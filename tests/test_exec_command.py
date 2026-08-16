"""Tests for the `agm exec` CLI command.

Covers:
- CLI wires FILE argument and params, --strict-json/--no-strict-json,
  --max-iters, --agent, --log-file, --no-log flags into ExecArgs
- Missing file exits with code 1 and prints to stderr
- Unreadable file exits with code 1 and prints error to stderr
- Valid programs execute through the program pipeline; static failures and uncaught
  AgL exceptions use their documented exit codes.
"""

from __future__ import annotations

import decimal
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import pytest
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
import agm.commands.exec as exec_command
from agm.cli_support.args import ExecArgs
from agm.commands import exec_program as exec_engine
from tests._agl_helpers import write_file_program


class RecordedArgs(Protocol):
    def __getattr__(self, name: str) -> object: ...


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def invoke(runner: CliRunner, argv: list[str]) -> Result:
    return runner.invoke(get_command(cli.app), argv, prog_name="agm", catch_exceptions=False)


def inline_args(command: str, *, param_tokens: list[str] | None = None) -> ExecArgs:
    """Build the ``ExecArgs`` an ``agm exec -c SOURCE`` invocation produces."""
    return ExecArgs(
        file=None,
        command=command,
        param_tokens=param_tokens or [],
        strict_json=None,
        max_iters=None,
        no_log=True,
        log_file=None,
    )


def file_args(path: Path) -> ExecArgs:
    """Build the ``ExecArgs`` an ``agm exec FILE`` invocation produces."""
    return ExecArgs(
        file=str(path),
        command=None,
        param_tokens=[],
        strict_json=None,
        max_iters=None,
        no_log=True,
        log_file=None,
    )


@pytest.fixture()
def recorded_runs(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Patch ``exec.run`` to record its ExecArgs instead of executing.

    Returns the list of recorded call arguments so parser-contract tests can
    assert how CLI flags map onto ``ExecArgs`` fields.
    """
    import agm.commands.exec as exec_mod

    calls: list[object] = []

    def fake_run(args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(exec_mod, "run", fake_run)
    return calls


class TestExecArgsParsing:
    """Parser-contract tests: verify CLI flags map to ExecArgs fields."""

    def test_exec_file_argument(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\n")

        result = invoke(runner, ["exec", str(agl_file)])
        assert result.exit_code == 0

        assert len(recorded_runs) == 1
        args = recorded_runs[0]
        assert getattr(args, "file") == str(agl_file)

    def test_exec_param_token_after_file(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\n")

        result = invoke(runner, ["exec", str(agl_file), "--k", "v"])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "param_tokens") == ["--k", "v"]

    def test_exec_multiple_param_tokens(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\n")

        result = invoke(runner, ["exec", str(agl_file), "--a", "1", "--b", "2"])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "param_tokens") == ["--a", "1", "--b", "2"]

    def test_exec_strict_json_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\n")

        result = invoke(runner, ["exec", "--strict-json", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "strict_json") is True

    def test_exec_no_strict_json_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\n")

        result = invoke(runner, ["exec", "--no-strict-json", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "strict_json") is False

    def test_exec_max_iters_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\n")

        result = invoke(runner, ["exec", "--max-iters", "10", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "max_iters") == 10

    def test_exec_agent_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\n")

        result = invoke(runner, ["exec", "--agent", 'AgentCommand("echo agent")', str(agl_file)])
        assert result.exit_code == 0

        assert getattr(recorded_runs[0], "agent") == 'AgentCommand("echo agent")'

    def test_exec_log_file_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\n")

        result = invoke(runner, ["exec", "--log-file", "/tmp/out.log", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "log_file") == "/tmp/out.log"

    def test_exec_no_log_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\n")

        result = invoke(runner, ["exec", "--no-log", str(agl_file)])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "no_log") is True

    def test_exec_program_flag(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "program def main() -> unit = ()\n")

        result = invoke(runner, ["exec", "-p", "main", str(agl_file)])
        assert result.exit_code == 0
        assert getattr(recorded_runs[0], "program") == "main"


class TestExecCommandArgParsing:
    """Parser-contract tests for the -c/--command option."""

    def test_exec_command_flag_maps_to_command(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["exec", "-c", 'print "hi"'])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "command") == 'print "hi"'
        assert getattr(args, "file") is None

    def test_exec_command_long_flag_maps_to_command(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["exec", "--command", "let x = 1"])
        assert result.exit_code == 0

        args = recorded_runs[0]
        assert getattr(args, "command") == "let x = 1"

    def test_exec_file_and_command_are_mutually_exclusive(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\n")
        result = invoke(runner, ["exec", "-c", "let x = 1", str(agl_file)])
        assert result.exit_code != 0
        # run() must not be reached when the CLI rejects the combination.
        assert recorded_runs == []

    def test_exec_neither_file_nor_command_exits_nonzero(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["exec"])
        assert result.exit_code != 0
        assert recorded_runs == []

    def test_exec_help_without_file_prints_help(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["exec", "--help"])
        assert result.exit_code == 0
        assert "agm exec" in result.output
        assert recorded_runs == []

    def test_exec_help_before_file_discovers_file_params(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "param msg: text\n")

        result = invoke(runner, ["exec", "--help", str(agl_file)])

        assert result.exit_code == 0
        assert "--msg" in result.output
        assert recorded_runs == []

    def test_exec_param_before_file_is_usage_error(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["exec", "--msg", "hello"])
        assert result.exit_code != 0
        assert "program parameter options must come after the FILE argument" in result.output
        assert recorded_runs == []

    def test_exec_inline_param_token_from_file_slot(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(runner, ["exec", "-c", "param msg\nprint msg", "--msg", "hello"])
        assert result.exit_code == 0
        args = recorded_runs[0]
        assert getattr(args, "file") is None
        assert getattr(args, "param_tokens") == ["--msg", "hello"]

    def test_exec_parser_preserves_canonical_qualified_param_flags(
        self, runner: CliRunner, recorded_runs: list[object]
    ) -> None:
        result = invoke(
            runner,
            [
                "exec",
                "workflow.agl",
                "--@module::no-settings::region",
                "local",
                "--no-@module::settings::region",
            ],
        )

        assert result.exit_code == 0
        args = recorded_runs[0]
        assert getattr(args, "param_tokens") == [
            "--@module::no-settings::region",
            "local",
            "--no-@module::settings::region",
        ]


class TestExecCommandInline:
    """Behavior tests for executing an inline -c/--command program."""

    def _command_args(self, command: str, *, param_tokens: list[str] | None = None) -> ExecArgs:
        return inline_args(command, param_tokens=param_tokens)

    def test_inline_command_runs_and_prints(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert exec_command.run(self._command_args('print "hello"')) is None
        assert capsys.readouterr().out == "hello\n"

    def test_inline_command_with_params(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = self._command_args("param msg\nprint msg", param_tokens=["--msg", "hi"])
        assert exec_command.run(args) is None
        assert capsys.readouterr().out == "hi\n"

    def test_inline_command_static_error_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(self._command_args("let x = undefined_name"))
        assert exc_info.value.code == 1
        assert capsys.readouterr().err

    def test_neither_file_nor_command_exits_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Calling run() with neither source set fails cleanly (defensive guard)."""
        args = ExecArgs(
            file=None,
            command=None,
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        assert "Error" in capsys.readouterr().err


class TestInlineSourceDiagnostics:
    """Diagnostics that only inline (`-c`) source can trigger explain themselves.

    Inline source without a ``program def`` is wrapped in a synthetic entry, so
    its top level is statement-oriented; declaring a ``program def`` suppresses
    the wrap and makes the same text an ordinary module with a static root. The
    static-root diagnostics say so, and ``resource``/``resource-dir`` explain
    that they have no module file to anchor against. A file entry keeps the
    plain wording.
    """

    def _failing_stderr(self, args: ExecArgs, capsys: pytest.CaptureFixture[str]) -> str:
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        return capsys.readouterr().err

    def test_inline_bare_expression_at_module_root_names_the_program_def(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = 'program def main() -> unit =\n  print "hi"\nprint "top"\n'
        assert "program def" in self._failing_stderr(inline_args(source), capsys)

    def test_inline_assignment_at_module_root_names_the_program_def(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = "var x = 1\nprogram def main() -> unit = print x\nx := 2\n"
        assert "program def" in self._failing_stderr(inline_args(source), capsys)

    def test_inline_non_constant_root_binding_names_the_program_def(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = "let x = 1 + 1\nprogram def main() -> unit = print x\n"
        assert "program def" in self._failing_stderr(inline_args(source), capsys)

    def test_inline_program_def_still_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert exec_command.run(inline_args('program def main() -> unit =\n  print "hi"\n')) is None
        assert capsys.readouterr().out == "hi\n"

    def test_file_bare_expression_at_module_root_keeps_the_plain_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        entry = tmp_path / "prog.agl"
        entry.write_text('program def main() -> unit =\n  print "hi"\nprint "top"\n')
        err = self._failing_stderr(file_args(entry), capsys)
        assert "static module root" in err
        assert "program def" not in err

    def test_file_assignment_at_module_root_keeps_the_plain_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        entry = tmp_path / "prog.agl"
        entry.write_text("var x = 1\nprogram def main() -> unit = print x\nx := 2\n")
        err = self._failing_stderr(file_args(entry), capsys)
        assert "static module root" in err
        assert "program def" not in err

    def test_file_non_constant_root_binding_keeps_the_plain_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        entry = tmp_path / "prog.agl"
        entry.write_text("let x = 1 + 1\nprogram def main() -> unit = print x\n")
        err = self._failing_stderr(file_args(entry), capsys)
        assert "constant expressions" in err
        assert "program def" not in err

    def test_inline_resource_call_explains_the_missing_module_file(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        err = self._failing_stderr(inline_args('print(resource("data.txt"))'), capsys)
        assert "resource" in err
        assert "inline" in err.lower()

    def test_file_resource_call_resolves_against_the_module_directory(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "data.txt").write_text("payload\n")
        entry = tmp_path / "prog.agl"
        entry.write_text('program def main() -> unit = print(resource("data.txt"))\n')
        assert exec_command.run(file_args(entry)) is None
        assert "data.txt" in capsys.readouterr().out


class TestExecDynamicHelp:
    def test_exec_help_for_file_includes_discovered_params(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'param msg: text = "hi"\nprogram def main() -> unit = print msg\n',
        )

        with pytest.raises(SystemExit) as exc_info:
            cli._exec_print_help(file=str(agl_file), command=None)

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Program parameters:" in out
        assert "--msg" in out

    def test_exec_help_uses_the_file_stem_for_colliding_entry_params(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "settings.agl").write_text('param region: text = "eu"\n')
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "import settings\nparam region: text\nprogram def main() -> unit = print region\n"
        )

        with pytest.raises(SystemExit) as exc_info:
            cli._exec_print_help(file=str(agl_file), command=None)

        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "--prog::region" in output
        assert "--<entry>::region" not in output

    def test_exec_accepts_the_file_stem_for_colliding_entry_params(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "settings.agl").write_text('param region: text = "eu"\n')
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text(
            "import settings\nparam region: text\nprogram def main() -> unit = print region\n"
        )

        assert (
            exec_command.run(_exec_args_no_log(agl_file, param_tokens=["--prog::region", "local"]))
            is None
        )
        assert capsys.readouterr().out == "local\n"

    def test_exec_help_for_inline_command_includes_discovered_params(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli._exec_print_help(
                file=None,
                command="param count: int = 1\nlet next = count + 1\nprint next",
            )

        assert exc_info.value.code == 0
        assert "--count" in capsys.readouterr().out

    def test_exec_help_ignores_qualified_option_bound_as_inline_file(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            cli._exec_print_help(
                file="--foo/bar::x",
                command="param count: int = 1\nprint count",
            )

        assert "--count" in capsys.readouterr().out

    def test_exec_help_discovers_params_from_cli_module_roots(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        module_root = tmp_path / "modules"
        module_root.mkdir()
        (module_root / "settings.agl").write_text('param region: text = "eu"\n')
        entry = tmp_path / "prog.agl"
        entry.write_text("import settings\nprogram def main() -> unit = ()\n")

        with pytest.raises(SystemExit):
            cli._exec_print_help(file=str(entry), command=None, module_paths=[str(module_root)])

        assert "--region" in capsys.readouterr().out

    def test_exec_help_discovers_params_from_configured_module_roots(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.config.context import ConfigContext

        module_root = tmp_path / "modules"
        module_root.mkdir()
        (module_root / "settings.agl").write_text('param region: text = "eu"\n')
        config_dir = tmp_path / ".agm"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(f'[modules]\nroots = ["{module_root}"]\n')
        monkeypatch.setattr(
            "agm.config.context.current_config_context",
            lambda: ConfigContext(home=tmp_path, proj_dir=None, cwd=tmp_path),
        )
        entry = tmp_path / "prog.agl"
        entry.write_text("import settings\nprogram def main() -> unit = ()\n")

        with pytest.raises(SystemExit):
            cli._exec_print_help(file=str(entry), command=None)

        assert "--region" in capsys.readouterr().out

    def test_exec_help_discovers_params_from_path_dependencies(
        self, tmp_path: Path, runner: CliRunner
    ) -> None:
        bravo = tmp_path / "bravo"
        (bravo / "bravo").mkdir(parents=True)
        (bravo / "package.toml").write_text('[package]\nname = "bravo"\nversion = "1.0.0"\n')
        (bravo / "bravo" / "settings.agl").write_text('param region: text = "eu"\n')
        alpha = tmp_path / "alpha"
        (alpha / "alpha").mkdir(parents=True)
        (alpha / "package.toml").write_text(
            '[package]\nname = "alpha"\nversion = "1.0.0"\n\n'
            "[dependencies]\n"
            'bravo = { version = "1", path = "../bravo" }\n'
        )
        entry = alpha / "alpha" / "main.agl"
        entry.write_text("import bravo/settings\nprogram def main() -> unit = ()\n")

        result = invoke(runner, ["exec", "--no-stdlib", str(entry), "--help"])

        assert result.exit_code == 0
        assert "--region" in result.output

    def test_exec_help_for_source_without_params_has_no_param_section(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli._exec_print_help(file=None, command='print "hi"')

        assert exc_info.value.code == 0
        assert "Program parameters:" not in capsys.readouterr().out

    def test_exec_help_degrades_when_effective_root_loading_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("param msg: text\n")
        monkeypatch.setattr(
            "agm.cli_support.exec_roots.effective_exec_roots",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("unavailable roots")),
        )

        with pytest.raises(SystemExit) as exc_info:
            cli._exec_print_help(file=str(agl_file), command=None)

        assert exc_info.value.code == 0
        assert "Program parameters:" not in capsys.readouterr().out

    def test_exec_help_for_unreadable_file_degrades(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli._exec_print_help(file=str(tmp_path / "missing.agl"), command=None)

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "agm exec" in out
        assert "Program parameters:" not in out


class TestExecCommandBehavior:
    """Behavior tests for the exec command run() function."""

    def test_missing_file_exits_1(self, tmp_path: Path) -> None:
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(tmp_path / "nonexistent.agl"),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1

    def test_missing_file_prints_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(tmp_path / "nonexistent.agl"),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit):
            exec_command.run(args)
        captured = capsys.readouterr()
        assert "Error" in captured.err or "error" in captured.err.lower()

    def test_unreadable_file_exits_1_with_friendly_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A path that is a directory (not a readable file) exits 1 with a friendly error."""
        from agm.cli_support.args import ExecArgs

        a_dir = tmp_path / "a_directory"
        a_dir.mkdir()

        args = ExecArgs(
            file=str(a_dir),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        # The friendly message names the offending path.
        assert "a_directory" in captured.err

    @pytest.mark.skipif(
        os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        reason="permission bits are meaningless for root or on non-POSIX platforms",
    )
    def test_an_existing_unreadable_file_containing_a_reference_separator_is_still_a_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An on-disk file wins classification even when its name contains
        ``::`` and it cannot be read: execution reports the ordinary
        unreadable-file error, never "does not name an active package" (the
        installed-reference error), matching the classification rule shared
        with ``--help`` and shell completion.
        """
        from agm.cli_support.args import ExecArgs

        unreadable = tmp_path / "pkg::mod"
        unreadable.write_text("param level: text\n", encoding="utf-8")
        unreadable.chmod(0)
        try:
            args = ExecArgs(
                file=str(unreadable),
                param_tokens=[],
                strict_json=None,
                max_iters=None,
                no_log=False,
                log_file=None,
            )
            with pytest.raises(SystemExit) as exc_info:
                exec_command.run(args)
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Error: cannot read" in captured.err
            assert "does not name an active package" not in captured.err
        finally:
            unreadable.chmod(0o644)

    def test_valid_file_exits_0_success(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A valid .agl file with no agent calls exits 0 (success)."""
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\nx\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        # a simple valid program succeeds
        result = exec_command.run(args)
        assert result is None  # returns None on success (exit 0)

    def test_static_error_file_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A .agl file with a static error exits 1 and prints diagnostics to stderr."""
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = undefined_name\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert captured.err

    def test_static_discovery_failure_does_not_truncate_trace(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = undefined_name\n")
        log_path = tmp_path / "trace.jsonl"
        log_path.write_text("existing trace\n", encoding="utf-8")

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=str(log_path),
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)

        assert exc_info.value.code == 1
        assert log_path.read_text(encoding="utf-8") == "existing trace\n"


def _exec_args(
    agl_file: Path, *, param_tokens: list[str] | None = None, log_file: str | None = None
) -> ExecArgs:
    """Build ExecArgs for *agl_file* with all optional flags defaulted."""
    return ExecArgs(
        file=str(agl_file),
        param_tokens=param_tokens or [],
        strict_json=None,
        max_iters=None,
        no_log=False,
        log_file=log_file,
    )


def _exec_args_no_log(agl_file: Path, **overrides: object) -> ExecArgs:
    """Build ``ExecArgs`` with trace logging disabled."""
    values = {"no_log": True, **overrides}
    return replace(_exec_args(agl_file), **values)


_skip_if_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="permission tests are meaningless as root (root bypasses file modes)",
)


class TestExecLogFileValidatedUpFront:
    """a non-writable --log-file fails up front with a clean Error + exit 1."""

    @_skip_if_root
    def test_unwritable_log_dir_exits_1_with_clean_error_before_running(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A --log-file under a read-only directory yields ``Error: ...`` + exit 1
        BEFORE any program statement runs (no raw PermissionError traceback)."""
        agl_file = tmp_path / "test.agl"
        # If the program ran, it would print to stdout — it must NOT.
        write_file_program(agl_file, 'print "should-not-run"\n')

        ro_dir = tmp_path / "ro"
        ro_dir.mkdir()
        ro_dir.chmod(0o555)
        log_path = ro_dir / "trace.log"

        try:
            with pytest.raises(SystemExit) as exc_info:
                exec_command.run(_exec_args(agl_file, log_file=str(log_path)))
        finally:
            ro_dir.chmod(0o755)  # restore so tmp_path cleanup succeeds

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        # Up-front failure: the program never ran, so no program output.
        assert "should-not-run" not in captured.out


class TestExecCommandEdgePaths:
    """Real-program coverage of the ok=True and pre-execution-error branches.

    The exit-2 (uncaught-AgL-exception) seam is exercised separately in
    ``TestExecExitCodeMapping`` because no real source reaches it.
    """

    def test_ok_result_returns_normally(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A successful real program prints its output and returns (exit 0)."""
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, 'print "ok"\n')

        # Real pipeline: no SystemExit on the success path.
        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert captured.out == "ok\n"

    def test_unknown_param_option_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, 'param msg: text = "ok"\nprint msg\n')

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file, param_tokens=["--unknown"]))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "error:" in captured.err

    def test_non_exhaustive_case_errors_and_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A non-exhaustive enum ``case`` fails statically before execution."""
        agl_file = tmp_path / "test.agl"
        write_file_program(
            agl_file,
            "enum R\n"
            "  | Pass\n"
            "  | Fail\n"
            "let r: R = Pass()\n"
            "case r of\n"
            '  | Pass() => print "passed"\n',
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "error:" in captured.err
        assert "Non-exhaustive" in captured.err
        assert "Fail" in captured.err


class TestExecExitCodeMapping:
    """the exit-2 (uncaught AgL exception) seam.

    Exit 2 is unreachable through current real source. These mocked tests pin the
    CLI's RunResult-to-exit mapping for uncaught AgL exceptions.
    """

    @pytest.mark.parametrize(
        ("fields", "expected_fragments"),
        [
            ({}, ["AgentParseError"]),
            (
                {"message": "could not parse agent output"},
                ["AgentParseError", "could not parse agent output"],
            ),
        ],
    )
    def test_uncaught_exception_maps_to_exit_2(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fields: dict[str, object],
        expected_fragments: list[str],
    ) -> None:
        from agm.agl.pipeline import PipelineDriver, RunError, RunResult

        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\nx\n")

        def fake_run(
            self: PipelineDriver,
            prepared: object,
            *,
            param_values: object = None,
            check_only: bool = False,
            **_kwargs: object,
        ) -> RunResult:
            if check_only:
                return RunResult(ok=True, diagnostics=[], error=None)
            return RunResult(
                ok=False,
                diagnostics=[],
                error=RunError(type_name="AgentParseError", fields=fields),
            )

        import agm.agl.pipeline as pipeline_mod

        monkeypatch.setattr(pipeline_mod.PipelineDriver, "run_prepared", fake_run)

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        for fragment in expected_fragments:
            assert fragment in captured.err


class TestExecCommandWarnings:
    """Warning-severity diagnostics are reported but never affect the exit code.

    These warning paths are driven through a mocked ``run_prepared`` that injects
    a warning diagnostic.
    The error→exit-1 path IS reachable through real source and is covered by
    ``test_error_diagnostic_still_exits_1`` below.
    """

    def test_warning_with_ok_returns_normally_and_prints_to_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Mocked warning: this pins that a
        # warning prints to stderr and never raises SystemExit (exit 0).
        from agm.agl.diagnostics import Diagnostic
        from agm.agl.pipeline import PipelineDriver, RunResult

        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\nx\n")

        warning = Diagnostic(
            message="declared agent 'reviewer' is unused",
            line=7,
            column=3,
            end_line=7,
            end_column=8,
            severity="warning",
        )

        def fake_run(
            self: PipelineDriver,
            prepared: object,
            *,
            param_values: object = None,
            check_only: bool = False,
            **_kwargs: object,
        ) -> RunResult:
            return RunResult(ok=True, diagnostics=[], error=None, warnings=[warning])

        import agm.agl.pipeline as pipeline_mod

        monkeypatch.setattr(pipeline_mod.PipelineDriver, "run_prepared", fake_run)

        # ok=True even with a warning: returns normally (exit 0).
        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert f"{agl_file}:7:3-7: warning: declared agent 'reviewer' is unused" in captured.err

    def test_error_diagnostic_still_exits_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Real source: an undefined name is a static (error-severity) diagnostic.
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = undefined_name\n")

        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "undefined_name" in captured.err
        assert captured.err.startswith("test.agl:2:11-24: error:")

    def test_inline_error_diagnostic_has_command_label(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Inline -c errors carry the ``<command>:`` source label (from SourceId)."""
        args = ExecArgs(
            file=None,
            command="let x = undefined_name\n",
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "undefined_name" in captured.err
        assert "1:9-22: error:" in captured.err
        # The graph loader stamps inline source with SourceId(label="<command>"),
        # so <command>: appears as the source label in the diagnostic output.
        assert "<command>:1:9-22: error:" in captured.err

    def test_warning_and_error_together_exits_1_and_prints_both(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Mocked: combining a warning with an error requires an organic warning,
        # which is injected here to pin that both print and exit is 1.
        from agm.agl.diagnostics import Diagnostic
        from agm.agl.pipeline import PipelineDriver, RunResult

        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\nx\n")

        warning = Diagnostic(
            message="unused binding",
            line=2,
            column=1,
            end_line=2,
            end_column=4,
            severity="warning",
        )
        error = Diagnostic(
            message="unknown name",
            line=5,
            column=9,
            end_line=5,
            end_column=13,
        )

        def fake_run(
            self: PipelineDriver,
            prepared: object,
            *,
            param_values: object = None,
            check_only: bool = False,
            **_kwargs: object,
        ) -> RunResult:
            if check_only:
                return RunResult(ok=True, diagnostics=[], error=None)
            return RunResult(ok=False, diagnostics=[error], error=None, warnings=[warning])

        import agm.agl.pipeline as pipeline_mod

        monkeypatch.setattr(pipeline_mod.PipelineDriver, "run_prepared", fake_run)

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert f"{agl_file}:2:1-3: warning: unused binding" in captured.err
        assert f"{agl_file}:5:9-12: error: unknown name" in captured.err
        assert f"{agl_file}:5:9-12: warning: unknown name" not in captured.err


class TestExecParsesSourceOnce:
    """``agm exec`` loads and scopes the graph exactly ONCE (no double parse).

    Regression guard: ``agm exec`` learns the declared-agent inventory (to wire
    registrations) AND executes the program.  Both must come from a single
    ``prepare_program`` call so the source is never loaded or scoped twice.
    """

    def test_exec_parses_and_scopes_source_exactly_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.agl.modules.loader as loader_mod
        import agm.agl.scope.program as scope_graph_mod
        from agm.agl.modules.ids import ModuleId
        from agm.agl.modules.loader import LoadedModule, ModuleGraph
        from agm.agl.modules.roots import RootSet
        from agm.agl.scope.program import ResolvedProgram
        from agm.agl.syntax.advisories import SpacedQualifier
        from agm.agl.syntax.nodes import Program
        from agm.core import dry_run

        agl_file = tmp_path / "prog.agl"
        # A declared+called agent: exec must read the inventory AND run the
        # static pipeline, the exact scenario that previously parsed twice.
        write_file_program(
            agl_file,
            'let impl = AgentCommand("impl")\nimpl.ask("do it")\n',
        )

        real_build_repl_graph = loader_mod.build_repl_graph
        real_resolve_program = scope_graph_mod.resolve_program
        build_graph_calls = 0
        resolve_program_calls = 0

        def counting_build_repl_graph(
            program: Program,
            next_start_id: int,
            *,
            path: Path | None,
            cached: dict[ModuleId, LoadedModule],
            roots: RootSet,
            default_stdlib: bool = True,
            spaced_qualifiers: tuple[SpacedQualifier, ...] = (),
            default_label: str = "<repl>",
            source_text: str = "",
        ) -> tuple[ModuleGraph, int, dict[ModuleId, LoadedModule]]:
            nonlocal build_graph_calls
            build_graph_calls += 1
            return real_build_repl_graph(
                program,
                next_start_id,
                path=path,
                cached=cached,
                roots=roots,
                default_stdlib=default_stdlib,
                spaced_qualifiers=spaced_qualifiers,
                default_label=default_label,
                source_text=source_text,
            )

        def counting_resolve_program(
            graph: ModuleGraph,
        ) -> ResolvedProgram:
            nonlocal resolve_program_calls
            resolve_program_calls += 1
            return real_resolve_program(graph)

        monkeypatch.setattr(loader_mod, "build_repl_graph", counting_build_repl_graph)
        monkeypatch.setattr(scope_graph_mod, "resolve_program", counting_resolve_program)
        # Dry-run drives the full static pipeline (parse → scope → typecheck →
        # reconcile) without executing any agent.
        monkeypatch.setattr(dry_run, "_ENABLED", True)

        assert exec_command.run(_exec_args(agl_file)) is None
        assert build_graph_calls == 1
        assert resolve_program_calls == 1


class TestExecLowersGraphOnce:
    """``agm exec`` lowers the module graph exactly ONCE per invocation.

    Regression guard: params are validated against the LOWERED program, so exec
    checks them before the trace file is prepared and only then executes.  Both
    steps must share a single lowering — a program must never pay for IR
    construction (and, under self-validation, IR validation) twice per run.
    """

    def _count_lowerings(self, monkeypatch: pytest.MonkeyPatch) -> list[object]:
        """Record one entry per ``lower_program`` call while still lowering for real."""
        import agm.agl.lower as lower_mod
        from agm.agl.ir.program import ExecutableProgram

        real_lower_program = lower_mod.lower_program
        lowerings: list[object] = []

        def counting_lower_program(*args: Any, **kwargs: Any) -> ExecutableProgram:
            executable = real_lower_program(*args, **kwargs)
            lowerings.append(executable)
            return executable

        monkeypatch.setattr(lower_mod, "lower_program", counting_lower_program)
        return lowerings

    def test_exec_lowers_graph_once_and_still_runs_the_program(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        lowerings = self._count_lowerings(monkeypatch)
        (tmp_path / "helper.agl").write_text('def greet(who: text) -> text = "hi %{who}"\n')
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file, 'open import helper\nparam who: text = "world"\nprint helper::greet(who)\n'
        )

        assert exec_command.run(_exec_args(agl_file, param_tokens=["--who", "agl"])) is None

        assert capsys.readouterr().out == "hi agl\n"
        assert len(lowerings) == 1

    def test_exec_dry_run_lowers_graph_once_and_reports_call_sites(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from agm.core import dry_run

        lowerings = self._count_lowerings(monkeypatch)
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'let impl = AgentCommand("impl")\nparam task: text = "do it"\nimpl.ask(task)\n',
        )
        monkeypatch.setattr(dry_run, "_ENABLED", True)

        assert exec_command.run(_exec_args(agl_file)) is None

        captured = capsys.readouterr()
        # --dry-run keeps its contract: the static call-site inventory is
        # reported and the program never executes.
        assert "call-sites:" in captured.out
        assert len(lowerings) == 1

    def test_param_error_exits_1_before_the_trace_file_is_prepared(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A param failure preempts the run: no trace file, no program output."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, 'param n: int\nprint "n=%{n}"\n')
        log_path = tmp_path / "trace.jsonl"

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file, log_file=str(log_path)))

        assert exc_info.value.code == 1
        assert capsys.readouterr().err
        assert not log_path.exists()


class TestExecCLIPaths:
    """Cover the CLI paths for missing FILE and --no-log/--log-file conflict."""

    def test_exec_missing_file_exits_nonzero(self, runner: CliRunner) -> None:
        result = invoke(runner, ["exec"])
        assert result.exit_code != 0

    def test_exec_no_log_and_log_file_conflict(
        self, runner: CliRunner, tmp_path: Path, recorded_runs: list[object]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\n")
        # recorded_runs intercepts exec.run so we don't actually run the file.
        result = invoke(
            runner,
            ["exec", "--no-log", "--log-file", "/tmp/x.log", str(agl_file)],
        )
        assert result.exit_code != 0


class TestExecCommandExitCodes:
    """Exec command exit codes for valid programs, params, and flag collisions."""

    def test_valid_program_exits_0(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = 1\nx\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        result = exec_command.run(args)
        assert result is None  # no SystemExit → exit 0

    def test_program_with_params_exits_0(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "param msg\nprint msg\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=["--msg", "hello"],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        result = exec_command.run(args)
        assert result is None

    def test_declared_agents_with_std_config_preserve_params(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(
            agl_file,
            'import std/config\nlet worker = AgentCommand("worker")\nstd/config::log := false\n'
            "param value: int\nprint value\n",
        )
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=["--value", "7"],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )

        assert exec_command.run(args) is None

    def test_missing_param_exits_1(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "param msg\nprint msg\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],  # missing 'msg'
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1

    def test_scoped_param_cli_flag_uses_its_full_path_spelling(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A scoped `param Deploy::region` is supplied as `--Deploy::region`."""
        agl_file = tmp_path / "test.agl"
        write_file_program(
            agl_file, 'scope Deploy\nparam region: text = "eu"\nend Deploy\nprint(Deploy::region)\n'
        )

        assert (
            exec_command.run(_exec_args(agl_file, param_tokens=["--Deploy::region", "prod"]))
            is None
        )

        assert capsys.readouterr().out == "prod\n"

    def test_scoped_param_config_key_uses_its_full_path_spelling(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A scoped param resolves from its qualified config table."""
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[test.Deploy]\nregion = "prod"\n')
        agl_file = tmp_path / "test.agl"
        write_file_program(
            agl_file,
            'scope Deploy\nparam region: text = "eu"\nend Deploy\n'
            "program def demo() -> unit = print(Deploy::region)\n",
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        assert capsys.readouterr().out == "prod\n"

    def test_param_default_can_read_a_static_binding(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(
            agl_file,
            "let base = 1\nparam value: int = base\nprogram def demo() -> unit = print value\n",
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        assert capsys.readouterr().out == "1\n"

    def test_param_default_can_read_a_mutable_static_binding(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(
            agl_file,
            "var base = 1\nparam value: int = base\nprogram def demo() -> unit = print value\n",
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        assert capsys.readouterr().out == "1\n"

    def test_param_default_function_preserves_lazy_static_binding(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(
            agl_file,
            "let base = 1\n"
            "def get_base() -> int = base\n"
            "param value: int = get_base()\n"
            "program def demo() -> unit = print base\n",
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        assert capsys.readouterr().out == "1\n"

    def test_legacy_params_section_does_not_supply_values(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[params]\nmsg = "legacy"\n')
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        agl_file = tmp_path / "test.agl"
        write_file_program(
            agl_file, 'param msg: text = "default"\nprogram def demo() -> unit = print msg\n'
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        assert capsys.readouterr().out == "default\n"

    def test_engine_key_named_param_is_rejected(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, 'param timeout: text = "30s"\nprint timeout\n')

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))

        assert exc_info.value.code == 1

    def test_undeclared_param_config_warns_but_runs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text("\n".join(["[test]", 'msgs = "ignored"']))
        agl_file = tmp_path / "test.agl"
        write_file_program(
            agl_file,
            'param msg: text = "ok"\nprogram def demo() -> unit = print msg\n',
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        # The program still runs on the declared default; the misspelled key is
        # reported so it is not silently dropped.
        assert captured.out == "ok\n"
        assert "msgs" in captured.err

    def test_ask_program_dispatches_through_the_value_dispatcher(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``agm exec`` dispatches the default ``Agent`` value at runtime."""
        from agm.agl.runtime.agents import AgentCallHostError
        from agm.cli_support.args import ExecArgs

        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, 'ask("hi")\n')

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=True,
            log_file=None,
        )

        # A host transport failure remains an in-language AgentCallError.
        def failing_agent(req: object) -> str:
            raise AgentCallHostError(
                cause="spawn_failure", exit_code=None, stderr_tail="no runner", elapsed=0.0
            )

        monkeypatch.setattr(exec_engine, "value_driven_agent_factory", lambda **_: failing_agent)
        monkeypatch.setattr(exec_engine, "create_agl_session_host", lambda **_: None)
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 2

    def test_dry_run_printing_program_exits_0_no_stdout(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``agm exec --dry-run`` runs the static pipeline only — no output."""
        from agm.cli_support.args import ExecArgs
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, 'print "hello"\n')

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        assert exec_command.run(args) is None  # exit 0
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_dry_run_static_error_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """a static-error program under --dry-run still exits 1."""
        from agm.cli_support.args import ExecArgs
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "let x = undefined_name\n")

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1

    def test_dry_run_unreachable_match_error_exits_1_before_execution(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agm.cli_support.args import ExecArgs
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'def dormant(x: bool) -> int =\n  case x of\n    | true => 1\nprint "unreachable"\n',
        )
        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert ": error:" in captured.err

    def test_static_error_exits_1_not_2(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "test.agl"
        write_file_program(agl_file, "let x = undefined_name\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1  # static error, not AgL exception


def _spy_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Patch ``exec.PipelineDriver`` with a recording subclass.

    Returns a dict that captures the constructor kwargs the command passed.
    """
    from agm.agl.pipeline import PipelineDriver as RealRuntime

    captured: dict[str, object] = {}

    class RecordingRuntime(RealRuntime):
        def __init__(
            self,
            *,
            default_loop_limit: int = 5,
            default_strict_json: bool = False,
            agent_dispatcher: Any | None = None,
            session_host: Any | None = None,
            shell_exec_timeout: float | None = None,
            default_call_depth_limit: int | None = None,
        ) -> None:
            captured["default_loop_limit"] = default_loop_limit
            captured["default_strict_json"] = default_strict_json
            captured["shell_exec_timeout"] = shell_exec_timeout
            captured["default_call_depth_limit"] = default_call_depth_limit
            super().__init__(
                default_loop_limit=default_loop_limit,
                default_strict_json=default_strict_json,
                agent_dispatcher=agent_dispatcher,
                session_host=session_host,
                shell_exec_timeout=shell_exec_timeout,
                default_call_depth_limit=default_call_depth_limit,
            )

    monkeypatch.setattr(exec_engine, "PipelineDriver", RecordingRuntime)
    return captured


class TestExecConfigWiring:
    """[exec] config (strict-json/max-iters) flows into the runtime."""

    def _config_home(self, tmp_path: Path) -> Path:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text("[exec]\nstrict-json = true\nmax-iters = 9\n")
        return home

    def test_config_values_reach_runtime_constructor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.cli_support.args import ExecArgs
        from agm.config.context import ConfigContext

        home = self._config_home(tmp_path)
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "let x = 1\nx\n")

        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        captured = _spy_runtime(monkeypatch)

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        assert exec_command.run(args) is None
        assert captured["default_strict_json"] is True
        assert captured["default_loop_limit"] == 9

    def test_cli_strict_json_overrides_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.cli_support.args import ExecArgs
        from agm.config.context import ConfigContext

        home = self._config_home(tmp_path)  # config sets strict_json = true
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "let x = 1\nx\n")

        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        captured = _spy_runtime(monkeypatch)

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=False,  # CLI --no-strict-json overrides config true
            max_iters=7,  # CLI --max-iters overrides config 9
            no_log=False,
            log_file=None,
        )
        assert exec_command.run(args) is None
        assert captured["default_strict_json"] is False
        assert captured["default_loop_limit"] == 7

    def test_timeout_config_flows_to_shell_exec_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """[exec] timeout config is wired to shell_exec_timeout on PipelineDriver."""
        from agm.cli_support.args import ExecArgs
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text("[exec]\ntimeout = 60\n")

        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "let x = 1\nx\n")

        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        captured = _spy_runtime(monkeypatch)

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        assert exec_command.run(args) is None
        assert captured["shell_exec_timeout"] == 60.0

    def test_invalid_timeout_config_exits_cleanly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: "pytest.CaptureFixture[str]",
    ) -> None:
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[exec]\ntimeout = "forever"\n')
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "let x = 1\n")

        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(
                ExecArgs(
                    file=str(agl_file),
                    param_tokens=[],
                    strict_json=None,
                    max_iters=None,
                    no_log=True,
                    log_file=None,
                )
            )

        assert exc_info.value.code == 1
        assert "Error: invalid exec configuration" in capsys.readouterr().err


def _exec_args_with_fallback_runtime(
    agl_file: Path, monkeypatch: pytest.MonkeyPatch, *, param_tokens: list[str] | None = None
) -> ExecArgs:
    """Return ExecArgs for *agl_file* and patch PipelineDriver to have a fallback agent.

    In real use the CLI wires the runner-backed default agent; in tests we
    patch the runtime to supply the default session host for free ``ask`` calls.
    """
    from agm.agl.pipeline import PipelineDriver as RealRuntime
    from agm.agl.runtime.agents import AgentFn
    from agm.agl.runtime.request import AgentRequest, AgentResponse

    def stub_agent(req: AgentRequest) -> AgentResponse:
        return AgentResponse(content="stub")

    class FallbackRuntime(RealRuntime):
        def __init__(
            self,
            *,
            default_loop_limit: int = 5,
            default_strict_json: bool = False,
            agent_dispatcher: AgentFn | None = None,
            session_host: Any | None = None,
            shell_exec_timeout: float | None = None,
            default_call_depth_limit: int | None = None,
        ) -> None:
            del agent_dispatcher
            super().__init__(
                default_loop_limit=default_loop_limit,
                default_strict_json=default_strict_json,
                agent_dispatcher=stub_agent,
                session_host=session_host,
                shell_exec_timeout=shell_exec_timeout,
                default_call_depth_limit=default_call_depth_limit,
            )

    monkeypatch.setattr(exec_engine, "PipelineDriver", FallbackRuntime)
    return _exec_args(agl_file, param_tokens=param_tokens)


class TestDryRunInventory:
    """--dry-run prints the ."""

    def test_dry_run_inventory_ask_call(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--dry-run prints one inventory entry per agent call site."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, 'let x = ask("Hello")\nx\n')

        args = _exec_args_with_fallback_runtime(agl_file, monkeypatch)
        assert exec_command.run(args) is None
        captured = capsys.readouterr()
        # Should print the call-sites inventory header and one entry.
        assert "call-sites" in captured.out
        assert "ask" in captured.out
        assert "text" in captured.out
        # The entry surfaces both the source line and column as "line N:C:"
        # (the captured call-site column is not dead).  `ask` starts at
        # column 9 of `let x = ask("Hello")`.
        assert "line 2:11:" in captured.out

    def test_dry_run_inventory_named_agent(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An Agent-method call appears in the inventory."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'let reviewer = AgentCommand("reviewer")\nreviewer.ask("Review this")\n',
        )

        args = _exec_args_with_fallback_runtime(agl_file, monkeypatch)
        assert exec_command.run(args) is None
        captured = capsys.readouterr()
        # Agent-method calls retain ``ask`` as the reported callee.
        assert "ask" in captured.out

    def test_dry_run_inventory_abort_policy(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit on_parse_error: abort policy surfaces in the inventory."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, 'ask("Hello", on_parse_error = Abort)\n')

        args = _exec_args_with_fallback_runtime(agl_file, monkeypatch)
        assert exec_command.run(args) is None
        captured = capsys.readouterr()
        assert "policy: abort" in captured.out

    def test_dry_run_inventory_no_call_sites_empty(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--dry-run with no agent calls produces no call-sites output."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, 'print "hello"\n')

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert "call-sites" not in captured.out

    def test_dry_run_inventory_static_error_exits_1_no_inventory(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Static error under --dry-run exits 1; no inventory is printed."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "let x = undefined_name\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args(agl_file))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "call-sites" not in captured.out

    def test_dry_run_inventory_nothing_executes(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--dry-run: the registered agent stub is never invoked."""
        from agm.agl.pipeline import PipelineDriver as RealRuntime
        from agm.agl.runtime.agents import AgentFn
        from agm.agl.runtime.request import AgentRequest, AgentResponse
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        agent_calls: list[AgentRequest] = []

        def spy_agent(req: AgentRequest) -> AgentResponse:
            agent_calls.append(req)
            raise AssertionError("agent should not be invoked in dry-run mode")

        class SpyRuntime(RealRuntime):
            def __init__(
                self,
                *,
                default_loop_limit: int = 5,
                default_strict_json: bool = False,
                agent_dispatcher: AgentFn | None = None,
                session_host: Any | None = None,
                shell_exec_timeout: float | None = None,
                default_call_depth_limit: int | None = None,
            ) -> None:
                del agent_dispatcher
                super().__init__(
                    default_loop_limit=default_loop_limit,
                    default_strict_json=default_strict_json,
                    agent_dispatcher=spy_agent,
                    session_host=session_host,
                    shell_exec_timeout=shell_exec_timeout,
                    default_call_depth_limit=default_call_depth_limit,
                )

        monkeypatch.setattr(exec_engine, "PipelineDriver", SpyRuntime)

        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, 'ask("Hi")\n')

        assert exec_command.run(_exec_args(agl_file)) is None
        assert agent_calls == []


class TestExecFFI:
    """``agm exec`` running a file-backed program that declares ``extern def``."""

    def test_exec_runs_an_extern_program_end_to_end(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "extern def add_one(x: int) -> int\nprint(add_one(41))\n")
        (tmp_path / "prog.py").write_text("def add_one(x):\n    return x + 1\n")

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert captured.out == "42\n"

    def test_dry_run_lists_the_extern_call_site_without_importing_companion(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--dry-run's inventory lists extern calls without companion side effects."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        marker = tmp_path / "marker.txt"
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "extern def add_one(x: int) -> int\nadd_one(41)\n")
        (tmp_path / "prog.py").write_text(
            f"open({str(marker)!r}, 'a').write('imported')\n"
            "def add_one(x):\n"
            f"    open({str(marker)!r}, 'a').write('called')\n"
            "    return x + 1\n"
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert "call-sites" in captured.out
        assert "add_one" in captured.out
        assert not marker.exists()

    def test_dry_run_skips_extern_import_and_execution(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A source engine-setting write must not import or call externs during --dry-run."""
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        marker = tmp_path / "marker.txt"
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "extern def choose_runner() -> text\nchoose_runner()\n")
        (tmp_path / "prog.py").write_text(
            f"open({str(marker)!r}, 'a').write('imported')\n"
            "def choose_runner():\n"
            f"    open({str(marker)!r}, 'a').write('called')\n"
            "    return 'echo'\n"
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert "call-sites" in captured.out
        assert "choose_runner" in captured.out
        assert not marker.exists()

    def test_dry_run_lists_extern_call_from_imported_module(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        marker = tmp_path / "marker.txt"
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "open import mylib\nmylib::run()\n")
        (tmp_path / "mylib.agl").write_text(
            "extern def from_lib(x: int) -> int\ndef run() -> int = from_lib(1)\n"
        )
        (tmp_path / "mylib.py").write_text(
            f"open({str(marker)!r}, 'a').write('imported')\ndef from_lib(x):\n    return x\n"
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert "from_lib" in captured.out
        assert not marker.exists()

    def test_dry_run_lists_extern_returned_from_ordinary_function(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        marker = tmp_path / "marker.txt"
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "extern def chosen(x: int) -> int\ndef choose() -> int -> int = chosen\nchoose()(1)\n",
        )
        (tmp_path / "prog.py").write_text(
            f"open({str(marker)!r}, 'a').write('imported')\ndef chosen(x):\n    return x\n"
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert "chosen" in captured.out
        assert not marker.exists()

    def test_dry_run_lists_extern_invoked_after_value_call_return(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from agm.core import dry_run

        monkeypatch.setattr(dry_run, "_ENABLED", True)

        marker = tmp_path / "marker.txt"
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "extern def chosen(x: int) -> int\n"
            "def get() -> int -> int = chosen\n"
            "let h = get\n"
            "h()(1)\n",
        )
        (tmp_path / "prog.py").write_text(
            f"open({str(marker)!r}, 'a').write('imported')\ndef chosen(x):\n    return x\n"
        )

        assert exec_command.run(_exec_args(agl_file)) is None
        captured = capsys.readouterr()
        assert "chosen" in captured.out
        assert "int -> int" not in captured.out
        assert not marker.exists()


class TestJsonParamsCLI:
    """--param with structured (record/array/decimal) types via JsonCodec."""

    def test_record_param_parsed_from_json_string(self, tmp_path: Path) -> None:
        """A record-typed param provided as a JSON string is parsed and usable."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "record Point\n  x: int\n  y: int\nparam pt: Point\nprint pt.x\n",
        )
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=['--pt={"x": 1, "y": 2}'],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        import io
        import sys

        out = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = out
        try:
            result = exec_command.run(args)
        finally:
            sys.stdout = old_stdout
        assert result is None
        assert out.getvalue().strip() == "1"

    def test_decimal_param_parsed_from_json_string(self, tmp_path: Path) -> None:
        """A decimal-typed param provided as a JSON string is accepted."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "param price: decimal\nprint price\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=["--price", "1.5"],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        import io
        import sys

        out = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = out
        try:
            result = exec_command.run(args)
        finally:
            sys.stdout = old_stdout
        assert result is None
        assert out.getvalue().strip() == "1.5"

    def test_array_param_parsed_from_json_string(self, tmp_path: Path) -> None:
        """An array-typed param provided as a JSON array string is accepted."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "param tags: array[text]\nprint tags\n")
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=['--tags=["a", "b"]'],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        import io
        import sys

        out = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = out
        try:
            result = exec_command.run(args)
        finally:
            sys.stdout = old_stdout
        assert result is None
        # The output should contain the rendered array.
        output = out.getvalue().strip()
        assert output  # non-empty

    def test_record_param_invalid_json_exits_1(self, tmp_path: Path) -> None:
        """A record-typed param with invalid JSON exits 1."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "record Point\n  x: int\n  y: int\nparam pt: Point\nprint pt.x\n",
        )
        from agm.cli_support.args import ExecArgs

        args = ExecArgs(
            file=str(agl_file),
            param_tokens=["--pt", "not_json"],
            strict_json=None,
            max_iters=None,
            no_log=False,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# uncaught-exception output includes source line/col
# ---------------------------------------------------------------------------


class TestUncaughtExceptionOutputFormat:
    """exec.py's exit-2 stderr includes the source location of runtime errors."""

    def _exec_args_nolog(self, agl_file: Path) -> "ExecArgs":
        from agm.cli_support.args import ExecArgs

        return ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=True,
            log_file=None,
        )

    def test_uncaught_exception_stderr_includes_line(
        self, tmp_path: Path, capsys: "pytest.CaptureFixture[str]"
    ) -> None:
        """Exit-2 stderr must include the source line number of the raise site."""
        agl_file = tmp_path / "prog.agl"
        # Force an uncaught ExecError from an exec call on line 1.
        write_file_program(agl_file, 'let x: int = exec "echo not-an-int"\nx\n')
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(self._exec_args_nolog(agl_file))
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        err = captured.err
        # The output must include a line reference (line 1).
        assert "line 2" in err or "line:2" in err or ":2:" in err, (
            f"Expected line reference in stderr, got: {err!r}"
        )


# ---------------------------------------------------------------------------
# Binary .agl file → clean error, exit 1
# ---------------------------------------------------------------------------


class TestExecBinaryFileError:
    """agm exec with a binary (non-UTF-8) .agl file exits 1 with clean error."""

    def test_binary_agl_file_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A binary file passed as the .agl source file exits 1 with a clean Error."""
        binary_file = tmp_path / "prog.agl"
        binary_file.write_bytes(b"\xff\xfe binary garbage \x00\x01\x02")

        args = ExecArgs(
            file=str(binary_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        # No raw traceback
        assert "UnicodeDecodeError" not in captured.err
        assert "Traceback" not in captured.err

    def test_binary_agl_file_no_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A binary .agl file must not produce any stdout before failing."""
        binary_file = tmp_path / "prog.agl"
        binary_file.write_bytes(b"\xff\xfe binary garbage \x00\x01\x02")

        args = ExecArgs(
            file=str(binary_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit):
            exec_command.run(args)
        captured = capsys.readouterr()
        assert captured.out == ""


def _install_marker_runner(directory: Path, env: dict[str, str], *, name: str, marker: str) -> Path:
    """Install a fake runner *name* that echoes *marker* plus the prompt-file path.

    The script prints two lines: the marker (identifying WHICH runner ran) and
    ``prompt-file=<path>`` (the prompt-file argument it received).  This lets a
    test assert both the resolved command and that ``%{PROMPT_FILE}`` / ``@file``
    substitution delivered a real path to the runner.
    """
    directory.mkdir(parents=True, exist_ok=True)
    runner = directory / name
    runner.write_text(
        "#!/bin/bash\n"
        f'echo "{marker}"\n'
        'for arg in "$@"; do\n'
        '  if [[ "$arg" == @* ]]; then\n'
        '    echo "prompt-file=${arg#@}"\n'
        '  elif [[ -f "$arg" ]]; then\n'
        '    echo "prompt-file=$arg"\n'
        "  fi\n"
        "done\n"
    )
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    if str(directory) not in env["PATH"].split(":"):
        env["PATH"] = str(directory) + ":" + env["PATH"]
    return runner


def _install_argv_echo_runner(
    directory: Path, env: dict[str, str], *, name: str, marker: str
) -> Path:
    """Install a fake runner *name* that echoes *marker* plus every raw argument.

    Unlike ``_install_marker_runner`` (which normalizes ``@file`` / existing-file
    arguments into a ``prompt-file=<path>`` line), this runner echoes each argv
    entry verbatim as ``arg=<raw>``.  That makes the difference between the
    ``%{PROMPT_FILE}`` placeholder branch (mid-argument substitution, e.g.
    ``--file=/abs/path``) and the bare-``@file`` append fallback (a separate
    trailing ``@/abs/path`` argument) observable in stdout.
    """
    directory.mkdir(parents=True, exist_ok=True)
    runner = directory / name
    runner.write_text(
        f'#!/bin/bash\necho "{marker}"\nfor arg in "$@"; do\n  echo "arg=$arg"\ndone\n'
    )
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    if str(directory) not in env["PATH"].split(":"):
        env["PATH"] = str(directory) + ":" + env["PATH"]
    return runner


class TestExecAgentValues:
    """Encoded Agent values select their own builders, not legacy runner maps."""

    def _run_agm_exec(
        self, args: list[str], *, env: dict[str, str], cwd: Path
    ) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            [sys.executable, "-m", "agm.cli", "exec", *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(cwd),
            check=False,
        )

    def test_agent_command_selects_its_own_runner(self, tmp_path: Path) -> None:
        env = dict(os.environ)
        env.setdefault("HOME", str(Path.home()))
        _install_marker_runner(tmp_path / "bin", env, name="value-runner", marker="FROM-VALUE")
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'let impl = AgentCommand("value-runner \\%{SESSION_ID}")\n'
            'let x = impl.ask("do it")\n'
            "print x\n",
        )
        result = self._run_agm_exec([str(agl_file), "--no-log"], env=env, cwd=tmp_path)

        assert result.returncode == 0, result.stderr
        assert "FROM-VALUE" in result.stdout

    def test_agent_command_preserves_prompt_file_substitution(self, tmp_path: Path) -> None:
        env = dict(os.environ)
        env.setdefault("HOME", str(Path.home()))
        _install_argv_echo_runner(tmp_path / "bin", env, name="value-runner", marker="FROM-VALUE")
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'let impl = AgentCommand("value-runner --file=\\%{PROMPT_FILE} \\%{SESSION_ID}")\n'
            'let x = impl.ask("do it")\nprint x\n',
        )

        result = self._run_agm_exec([str(agl_file), "--no-log"], env=env, cwd=tmp_path)

        assert result.returncode == 0, result.stderr
        assert re.search(r"^arg=--file=/", result.stdout, re.MULTILINE)
        assert "arg=@/" not in result.stdout


class TestExecTimeoutAndLogFileFlags:
    """CLI ``--timeout`` / ``--no-timeout`` / ``--no-log-file`` resolution."""

    def _capture_timeout(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        from collections.abc import Mapping

        from agm.agl.ir.ids import SymbolId
        from agm.agl.ir.program import ExecutableProgram
        from agm.agl.matchcompile import MatchCompiledProgram
        from agm.agl.pipeline import PipelineDriver as RealRuntime
        from agm.agl.pipeline import PreparedProgram, RunResult
        from agm.agl.runtime.host_settings import HostSettingsPolicy
        from agm.agl.semantics.values import Value

        captured: dict[str, object] = {}

        class CapturingRuntime(RealRuntime):
            def run_prepared(
                self,
                prepared: PreparedProgram,
                *,
                param_values: Mapping[str, object] | None = None,
                check_only: bool = False,
                log_file: Path | None = None,
                compiled: MatchCompiledProgram | None = None,
                executable: ExecutableProgram | None = None,
                host_settings_policy: HostSettingsPolicy | None = None,
                builtin_host_settings: Mapping[str, Value] | None = None,
                program_symbol: SymbolId | None = None,
            ) -> RunResult:
                captured["shell_exec_timeout"] = self._shell_exec_timeout
                return super().run_prepared(
                    prepared,
                    param_values=param_values,
                    check_only=check_only,
                    log_file=log_file,
                    compiled=compiled,
                    executable=executable,
                    host_settings_policy=host_settings_policy,
                    builtin_host_settings=builtin_host_settings,
                    program_symbol=program_symbol,
                )

        monkeypatch.setattr(exec_engine, "PipelineDriver", CapturingRuntime)
        return captured

    def test_cli_timeout_flag_sets_engine_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "let x = 1\nx\n")
        captured = self._capture_timeout(monkeypatch)

        result = exec_command.run(_exec_args_no_log(agl_file, timeout="45s"))
        assert result is None
        assert captured["shell_exec_timeout"] == pytest.approx(45.0)

    def test_cli_invalid_timeout_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "let x = 1\nx\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file, timeout="not-a-duration"))
        assert exc_info.value.code == 1
        assert "invalid --timeout" in capsys.readouterr().err

    def test_cli_no_timeout_clears_engine_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "let x = 1\nx\n")
        captured = self._capture_timeout(monkeypatch)

        result = exec_command.run(_exec_args_no_log(agl_file, no_timeout=True))
        assert result is None
        assert captured["shell_exec_timeout"] is None

    def test_cli_timeout_preserves_raw_builtin_value(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "import std/config\nprint std/config::timeout\n")

        exec_command.run(_exec_args_no_log(agl_file, timeout="0.0001s"))

        assert capsys.readouterr().out == 'Option::Some(value = "0.0001s")\n'

    def test_program_config_timeout_preserves_raw_builtin_value(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[exec]\ntimeout = "1s"\n\n[prog.main]\ntimeout = 0.0001\n'
        )
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "import std/config\nprint std/config::timeout\n")
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        exec_command.run(_exec_args_no_log(agl_file))

        assert capsys.readouterr().out == 'Option::Some(value = "0.0001s")\n'

    def test_cli_no_log_file_clears_visible_seed_not_configured_trace(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from agm.config.context import ConfigContext

        trace_path = tmp_path / "configured.jsonl"
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            f'[exec]\nlog = true\nlog-file = "{trace_path}"\n'
        )
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "import std/config\nprint std/config::log-file\n")
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        exec_command.run(_exec_args_no_log(agl_file, no_log=False, no_log_file=True))

        assert capsys.readouterr().out == "Option::None\n"
        assert trace_path.exists()


class TestExecSourceConfigPrecedence:
    """Source engine-setting declarations resolved by ``agm exec``.

    A source ``std/config::KEY := VALUE`` assignment takes effect from its
    program point onward and overrides the CLI flag seed; the config-file layer
    is the floor.  Each test uses behavioral assertions — observable exit codes
    and output — rather than internal call counts, following the testing policy.
    """

    # ------------------------------------------------------------------
    # max_iters source declaration
    # ------------------------------------------------------------------

    def test_source_max_iters_caps_loop_at_source_value(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``std/config::max-iters := 3`` in source caps the do loop at 3 iterations.

        The loop ``until n >= 100`` cannot complete in 3 iterations (n starts at
        0 and increments by 1), so the runtime raises a LoopLimitExceeded and
        the command exits 2.
        """
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\n"
            "std/config::max-iters := 3\n"
            "var n = 0\n"
            "do\n"
            "  n := n + 1\n"
            "until n >= 100\n",
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))
        assert exc_info.value.code == 2

    def test_source_max_iters_allows_completion_when_sufficient(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``std/config::max-iters := 100`` allows a do loop needing exactly 100 iterations."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\n"
            "std/config::max-iters := 100\n"
            "var n = 0\n"
            "do\n"
            "  n := n + 1\n"
            "until n >= 100\n"
            'print "done"\n',
        )
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None  # exit 0
        assert capsys.readouterr().out == "done\n"

    def test_source_max_iters_zero_disables_host_limit(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``std/config::max-iters := 0`` turns an active host valve off."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\n"
            "std/config::max-iters := 0\n"
            "var i = 0\n"
            "do\n"
            "  i := i + 1\n"
            "until i >= 3\n"
            "print i\n",
        )

        exec_command.run(_exec_args_no_log(agl_file, max_iters=1))
        assert capsys.readouterr().out == "3\n"

    def test_source_max_iters_negative_expression_is_rejected(self, tmp_path: Path) -> None:
        """A computed negative ``max-iters`` value is rejected at the assignment point."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file, "import std/config\nlet bad = 0 - 1\nstd/config::max-iters := bad\nprint 1\n"
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))
        assert exc_info.value.code == 2

    def test_source_max_iters_overrides_cli_max_iters(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Source ``std/config::max-iters := 3`` overrides CLI ``--max-iters 100``.

        The CLI ``--max-iters 100`` seeds the loop valve, but the source
        assignment takes effect from its program point and caps the loop at 3.
        The loop needs 100 iterations, so it exceeds the source cap and the
        command exits 2 (source wins over the CLI flag).
        """
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\n"
            "std/config::max-iters := 3\n"
            "var n = 0\n"
            "do\n"
            "  n := n + 1\n"
            "until n >= 100\n"
            'print "done"\n',
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file, max_iters=100))
        assert exc_info.value.code == 2  # source 3 overrides CLI 100

    def test_source_max_iters_overrides_config_max_iters(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Source ``std/config::max-iters := 100`` overrides ``[exec] max-iters = 3`` in config.

        Config says 3 (loop would fail); source assignment says 100 (loop completes).
        """
        from agm.config.general import ExecConfig

        low_limit_config = ExecConfig(
            strict_json=False,
            default_loop_limit=3,
            timeout=None,
            log=False,
            log_file=None,
        )
        monkeypatch.setattr(
            exec_engine, "exec_config_from_merged", lambda *_, **__: low_limit_config
        )

        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\n"
            "std/config::max-iters := 100\n"
            "var n = 0\n"
            "do\n"
            "  n := n + 1\n"
            "until n >= 100\n"
            'print "done"\n',
        )
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None  # exit 0 — source 100 overrides config 3
        assert capsys.readouterr().out == "done\n"

    # ------------------------------------------------------------------
    # max-iters valve scope: self-bounded loops are exempt
    # ------------------------------------------------------------------

    def test_max_iters_does_not_cap_for_over_finite_collection(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--max-iters`` caps only unguarded loops; a ``for`` over a finite
        collection larger than the cap must run to completion.

        Regression: the valve applied to all loops, so ``--max-iters 3`` broke
        ``for x in [1,2,3,4]``.
        """
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "var s = 0\nfor x in [1, 2, 3, 4, 5] do s := s + x done\nprint s\n",
        )
        result = exec_command.run(_exec_args_no_log(agl_file, max_iters=3))
        assert result is None  # exit 0
        assert capsys.readouterr().out == "15\n"

    def test_max_iters_does_not_cap_bounded_do_n_loop(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--max-iters`` does not cap a ``do[n]`` loop whose own bound exceeds it.

        The loop's own ``[n]`` bound is its termination machinery; the host
        safety valve must not cut it short.
        """
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "var i = 0\ndo[10]\n  i := i + 1\nuntil i >= 5\nprint i\n")
        result = exec_command.run(_exec_args_no_log(agl_file, max_iters=3))
        assert result is None  # exit 0
        assert capsys.readouterr().out == "5\n"

    def test_max_iters_caps_unbounded_do_until_loop(self, tmp_path: Path) -> None:
        """``--max-iters`` caps an unguarded ``do…until`` loop (no [n], no for)."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "var i = 0\ndo\n  i := i + 1\nuntil i >= 1000\nprint i\n")
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file, max_iters=3))
        assert exc_info.value.code == 2

    @pytest.mark.parametrize("limit", [0, -1])
    def test_cli_max_iters_requires_positive_value(self, tmp_path: Path, limit: int) -> None:
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "print 1\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file, max_iters=limit))

        assert exc_info.value.code == 1

    def test_max_iters_five_enables_valve(self, tmp_path: Path) -> None:
        """An explicit positive CLI limit consistently enables the valve."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "var i = 0\ndo\n  i := i + 1\nuntil i >= 1000\nprint i\n")
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file, max_iters=5))
        assert exc_info.value.code == 2

    # ------------------------------------------------------------------
    # strict-json source declaration
    # ------------------------------------------------------------------

    def test_source_strict_json_flows_into_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``std/config::strict-json := true`` in source does NOT pre-fold into the
        PipelineDriver constructor; it is applied when the assignment executes.
        The constructor receives the config-file value (False by default)."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\nstd/config::strict-json := true\nlet x = 1\nx\n",
        )

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None
        # constructor gets the config-file default; the source assignment
        # applies the live change at the point where it executes.
        assert captured["default_strict_json"] is False

    def test_source_strict_json_overrides_cli_strict_json(self, tmp_path: Path) -> None:
        """Source ``std/config::strict-json := true`` overrides CLI ``--no-strict-json``.

        The CLI ``--no-strict-json`` seeds the lenient floor, but the source
        assignment flips strict JSON on before the ``exec`` parses a fenced JSON
        result.  Under strict parsing the fenced payload is rejected, so the
        command exits 2 (source wins over the CLI flag; under the lenient CLI
        floor alone the value would have parsed and the command exited 0).
        """
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\n"
            "std/config::strict-json := true\n"
            "let r: int = exec \"printf '```json\\n5\\n```'\"\n"
            "print r\n",
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file, strict_json=False))
        assert exc_info.value.code == 2

    def test_source_strict_json_overrides_config_strict_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Source ``std/config::strict-json := false`` overrides ``[exec] strict-json = true``
        when the assignment executes. The PipelineDriver constructor still receives
        the config-file value (True)."""
        from agm.config.general import ExecConfig

        strict_config = ExecConfig(
            strict_json=True,
            default_loop_limit=5,
            timeout=None,
            log=False,
            log_file=None,
        )
        monkeypatch.setattr(exec_engine, "exec_config_from_merged", lambda *_, **__: strict_config)

        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            "import std/config\nstd/config::strict-json := false\nlet x = 1\nx\n",
        )

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None
        # constructor gets config-file value (True); the source assignment (False)
        # overrides it at runtime when it executes.
        assert captured["default_strict_json"] is True

    # ------------------------------------------------------------------
    # timeout source declaration
    # ------------------------------------------------------------------

    def test_source_timeout_flows_into_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``std/config::timeout := Some("30s")`` in source does NOT pre-fold into the
        PipelineDriver constructor; it is applied when the assignment executes.
        The constructor receives the config-file value (None by default)."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'import std/config\nstd/config::timeout := Some("30s")\nlet x = 1\nx\n',
        )

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None
        # constructor gets the config-file default (None); the source
        # declaration updates shell_exec_timeout at binding time.
        assert captured["shell_exec_timeout"] is None

    def test_source_timeout_integer_rejected(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``std/config::timeout := 60`` (integer) is a type error: timeout is Option[text]."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, "import std/config\nstd/config::timeout := 60\nlet x = 1\nx\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))
        assert exc_info.value.code == 1

    def test_source_timeout_overrides_config_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Source ``std/config::timeout := Some("30s")`` overrides ``[exec] timeout = 999``
        when the assignment executes. The PipelineDriver constructor still receives
        the config-file value (999.0)."""
        from agm.config.general import ExecConfig

        config_with_timeout = ExecConfig(
            strict_json=False,
            default_loop_limit=5,
            timeout=999.0,
            log=False,
            log_file=None,
        )
        monkeypatch.setattr(
            exec_engine, "exec_config_from_merged", lambda *_, **__: config_with_timeout
        )

        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file,
            'import std/config\nstd/config::timeout := Some("30s")\nlet x = 1\nx\n',
        )

        captured = _spy_runtime(monkeypatch)
        result = exec_command.run(_exec_args_no_log(agl_file))
        assert result is None
        # constructor gets config-file value (999.0); the source assignment (30s)
        # overrides it at runtime when it executes.
        assert captured["shell_exec_timeout"] == pytest.approx(999.0)

    def test_source_timeout_invalid_string_raises_runtime_error(self, tmp_path: Path) -> None:
        """A source timeout string that type-checks but fails parse_timeout raises
        a clean AgL-level ValueError at the assignment point (exit 2).

        ``std/config::timeout := Some("forever")`` is a valid ``Option[text]`` value
        so it passes scope and typecheck; the setting-write handler converts the
        parse_timeout ValueError to an AglRaise (uncaught AgL exception → exit 2).
        """
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file, 'import std/config\nstd/config::timeout := Some("forever")\nlet x = 1\nx\n'
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))
        assert exc_info.value.code == 2

    # ------------------------------------------------------------------
    # log source declaration
    # ------------------------------------------------------------------

    def test_source_log_write_creates_trace_file(self, tmp_path: Path) -> None:
        """``std/config::log := true`` in source enables trace logging (creates a file)."""
        agl_file = tmp_path / "prog.agl"
        write_file_program(agl_file, 'import std/config\nstd/config::log := true\nprint "hi"\n')

        # Run in tmp_path so .agent-files/ is created there.
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            exec_command.run(
                ExecArgs(
                    file=str(agl_file),
                    param_tokens=[],
                    strict_json=None,
                    max_iters=None,
                    no_log=False,
                    log_file=None,
                )
            )
        finally:
            os.chdir(old_cwd)

        agent_files = tmp_path / ".agent-files"
        log_files = list(agent_files.glob("exec-*.jsonl"))
        assert log_files, "Expected a trace log file from std/config::log := true"

    def test_source_log_file_write_writes_to_specified_path(self, tmp_path: Path) -> None:
        """``std/config::log-file := Some("path")`` in source writes the trace to that path."""
        log_path = tmp_path / "trace.log"
        agl_file = tmp_path / "prog.agl"
        write_file_program(
            agl_file, f'import std/config\nstd/config::log-file := Some("{log_path}")\nprint "hi"\n'
        )

        exec_command.run(
            ExecArgs(
                file=str(agl_file),
                param_tokens=[],
                strict_json=None,
                max_iters=None,
                no_log=False,
                log_file=None,
            )
        )
        assert log_path.exists(), "Expected trace log at source-specified path"


def _exec_args_inline_no_log(
    command: str,
    *,
    strict_json: bool | None = None,
    max_iters: int | None = None,
) -> ExecArgs:
    """Build a minimal ExecArgs for -c inline exec tests."""
    return ExecArgs(
        file=None,
        command=command,
        param_tokens=[],
        strict_json=strict_json,
        max_iters=max_iters,
        no_log=True,
        log_file=None,
        log=False,
    )


class TestExecModuleRoots:
    """``agm exec`` uses program pipeline and module roots.

    Tests verify that:
    - ``agm exec <file>`` uses the file's directory as invocation root.
    - ``agm exec -c`` uses cwd as invocation root.
    - An error in an imported module carries that module's file path in the
      diagnostic (via source_label).
    - A multi-file program executes successfully when the library is reachable.
    """

    def test_exec_file_uses_file_directory_as_root(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``agm exec file.agl`` can import a sibling module in the same directory."""
        lib_dir = tmp_path
        (lib_dir / "mylib.agl").write_text("def answer() -> int = 42\n")
        entry = lib_dir / "entry.agl"
        write_file_program(entry, "open import mylib\nlet r = answer()\nprint r\n")

        # A successful run returns normally (no SystemExit).
        exec_command.run(_exec_args_no_log(entry))
        captured = capsys.readouterr()
        assert "42" in captured.out

    def test_exec_file_import_error_reports_source_label(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Error in imported module shows that module's file path in diagnostic."""
        lib_dir = tmp_path
        (lib_dir / "broken.agl").write_text("def f() -> int = undeclared_name\n")
        entry = lib_dir / "entry.agl"
        write_file_program(entry, "open import broken\nlet r = f()\nr\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(entry))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        # The diagnostic should mention the broken.agl file path
        assert "broken.agl" in captured.err

    def test_exec_inline_uses_cwd_as_root(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``agm exec -c`` uses cwd (from config context) as invocation root."""
        # Write a lib module to a tmp dir
        lib_dir = tmp_path / "libdir"
        lib_dir.mkdir()
        (lib_dir / "util.agl").write_text('def greet() -> text = "Hi!"\n')
        entry_source = "open import util\nlet r = greet()\nprint r\n"

        # Patch current_config_context as imported in exec_command
        from agm.config import context as ctx_mod

        original_ctx = ctx_mod.current_config_context()

        class FakeCtx:
            home = original_ctx.home
            proj_dir = original_ctx.proj_dir
            cwd = lib_dir

        monkeypatch.setattr(exec_engine, "current_config_context", lambda: FakeCtx())

        # A successful run returns normally (no SystemExit).
        exec_command.run(_exec_args_inline_no_log(entry_source))
        captured = capsys.readouterr()
        assert "Hi!" in captured.out

    def test_exec_file_missing_import_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A missing import causes exit 1 with a diagnostic on stderr."""
        entry = tmp_path / "prog.agl"
        write_file_program(entry, "open import no_such_module\nlet x = 1\nx\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(entry))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "no_such_module" in captured.err

    def test_exec_multifile_successful_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A two-module AgL program executes successfully end-to-end."""
        lib_dir = tmp_path
        (lib_dir / "calc.agl").write_text("def square(n: int) -> int = n * n\n")
        entry = lib_dir / "prog.agl"
        write_file_program(entry, "open import calc\nlet r = square(4)\nprint r\n")

        # A successful run returns normally (no SystemExit).
        exec_command.run(_exec_args_no_log(entry))
        captured = capsys.readouterr()
        assert "16" in captured.out

    def test_invalid_module_roots_config_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ValueError from load_module_roots causes exit 1 with a config error."""
        from unittest.mock import patch

        entry = tmp_path / "prog.agl"
        write_file_program(entry, "let x = 1\nx\n")

        with (
            patch(
                "agm.cli_support.exec_roots.load_module_roots",
                side_effect=ValueError("bad config"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            exec_command.run(_exec_args_no_log(entry))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "module roots" in captured.err.lower()

    def test_stdlib_version_mismatch_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stdlib version mismatch from root resolution causes exit 1."""
        from agm.config.module_roots import StdlibVersionMismatchError

        entry = tmp_path / "prog.agl"
        write_file_program(entry, "let x = 1\nx\n")

        def fake_resolve_stdlib_root(*, home: Path) -> Path:
            raise StdlibVersionMismatchError("0.0.1", "0.1.0")

        monkeypatch.setattr(
            "agm.cli_support.exec_roots.resolve_stdlib_root", fake_resolve_stdlib_root
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(entry))
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "0.0.1" in captured.err
        assert "just install" in captured.err

    def test_configured_lib_root_is_resolved(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A configured lib_root in module roots config is resolved and used."""
        from agm.config.module_roots import ModuleRootsConfig

        lib_dir = tmp_path / "mylib"
        lib_dir.mkdir()
        (lib_dir / "shared.agl").write_text("def pi() -> int = 314\n")

        entry_dir = tmp_path / "work"
        entry_dir.mkdir()
        entry = entry_dir / "prog.agl"
        write_file_program(entry, "open import shared\nlet r = pi()\nprint r\n")

        # Patch load_module_roots to return a ModuleRootsConfig with lib_root set.
        with monkeypatch.context() as mp:
            mp.setattr(
                "agm.cli_support.exec_roots.load_module_roots",
                lambda *, home, proj_dir, cwd: ModuleRootsConfig(
                    lib_root=(str(lib_dir), tmp_path),  # absolute path
                    extra=(),
                ),
            )
            exec_command.run(_exec_args_no_log(entry))
        captured = capsys.readouterr()
        assert "314" in captured.out

    def test_exec_wildcard_import_multifile(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``open import pkg/*`` imports two sibling modules and makes both callable.

        Verifies that the wildcard import path works end-to-end through the
        exec_command pipeline (discover_params + run_prepared).
        """
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "add.agl").write_text("def add(a: int, b: int) -> int = a + b\n")
        (pkg_dir / "mul.agl").write_text("def mul(a: int, b: int) -> int = a * b\n")
        entry = tmp_path / "prog.agl"
        write_file_program(
            entry, "open import pkg/*\nlet s = add(3, 4)\nlet p = mul(3, 4)\nprint s\nprint p\n"
        )

        exec_command.run(_exec_args_no_log(entry))
        captured = capsys.readouterr()
        assert "7" in captured.out
        assert "12" in captured.out

    def test_exec_qualified_import_multifile(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``import x`` + ``x::name`` call works end-to-end.

        Verifies that the import path works through the command pipeline.
        """
        (tmp_path / "mathlib.agl").write_text("def square(n: int) -> int = n * n\n")
        entry = tmp_path / "prog.agl"
        write_file_program(entry, "import mathlib\nlet r = mathlib::square(7)\nprint r\n")

        exec_command.run(_exec_args_no_log(entry))
        captured = capsys.readouterr()
        assert "49" in captured.out

    def test_exec_imported_function_ask_multi_scenario(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multi-scenario: imported function calls ask() with an agent passed from entry.

        Scenario A: agent returns "Alice" → result printed is "Alice"
        Scenario B: agent returns "World" → result printed is "World"

        Each scenario drives exec_command.run end-to-end with a distinct mock
        response injected via runner_backed_agent_factory, asserting each output.
        """
        from agm.agl.runtime.request import AgentRequest, AgentResponse

        (tmp_path / "greeter.agl").write_text(
            "def greet(prompt: text, bot: Agent) -> text =\n  bot.ask(prompt)\n"
        )
        entry = tmp_path / "entry.agl"
        write_file_program(
            entry,
            "open import greeter\n"
            'let mybot = AgentCommand("mock")\n'
            'let result = greeter::greet("What is your name?", mybot)\n'
            "print result\n",
        )

        def _run_with_response(response: str) -> str:
            """Inject *response* as the mock agent answer and return stdout."""

            def mock_agent(req: AgentRequest) -> AgentResponse:
                return AgentResponse(content=response)

            monkeypatch.setattr(exec_engine, "value_driven_agent_factory", lambda **_: mock_agent)
            monkeypatch.setattr(exec_engine, "create_agl_session_host", lambda **_: None)
            exec_command.run(_exec_args_no_log(entry))
            out, _ = capsys.readouterr()
            return out

        # Scenario A: agent returns "Alice"
        out_a = _run_with_response("Alice")
        assert "Alice" in out_a

        # Scenario B: agent returns "World" — different input/output combination
        out_b = _run_with_response("World")
        assert "World" in out_b
        assert out_a != out_b


class TestExecCliModulePaths:
    """``-I/--module-path`` roots are threaded into ``assemble_roots``.

    Tests verify that a module placed only in a ``-I DIR`` root is resolvable
    by ``agm exec`` when that root is passed via ``ExecArgs.module_paths``.
    """

    def test_module_in_cli_root_is_resolvable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A module placed only in a -I root is importable by the entry program."""
        lib_root = tmp_path / "mylibs"
        lib_root.mkdir()
        (lib_root / "helper.agl").write_text("def answer() -> int = 99\n")

        # entry.agl lives in a separate directory with no sibling modules
        entry_dir = tmp_path / "prog"
        entry_dir.mkdir()
        entry = entry_dir / "main.agl"
        write_file_program(entry, "open import helper\nlet r = answer()\nprint r\n")

        args = ExecArgs(
            file=str(entry),
            command=None,
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=True,
            log_file=None,
            log=False,
            module_paths=[str(lib_root)],
        )
        exec_command.run(args)
        captured = capsys.readouterr()
        assert "99" in captured.out

    def test_multiple_cli_roots_each_resolvable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Two -I roots each contribute a distinct module, both resolvable."""
        root_a = tmp_path / "rootA"
        root_a.mkdir()
        (root_a / "mod_a.agl").write_text("def va() -> int = 10\n")

        root_b = tmp_path / "rootB"
        root_b.mkdir()
        (root_b / "mod_b.agl").write_text("def vb() -> int = 20\n")

        entry_dir = tmp_path / "entry"
        entry_dir.mkdir()
        entry = entry_dir / "prog.agl"
        write_file_program(
            entry,
            "open import mod_a\nopen import mod_b\nlet r = va() + vb()\nprint r\n",
        )

        args = ExecArgs(
            file=str(entry),
            command=None,
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=True,
            log_file=None,
            log=False,
            module_paths=[str(root_a), str(root_b)],
        )
        exec_command.run(args)
        captured = capsys.readouterr()
        assert "30" in captured.out

    def test_module_not_found_without_cli_root(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without -I, a module in an external root is not found (exit 1)."""
        lib_root = tmp_path / "mylibs"
        lib_root.mkdir()
        (lib_root / "helper.agl").write_text("def answer() -> int = 99\n")

        entry_dir = tmp_path / "prog"
        entry_dir.mkdir()
        entry = entry_dir / "main.agl"
        write_file_program(entry, "open import helper\nlet r = answer()\nprint r\n")

        args = ExecArgs(
            file=str(entry),
            command=None,
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=True,
            log_file=None,
            log=False,
            module_paths=[],  # no CLI roots
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "helper" in captured.err

    def test_inline_exec_with_cli_root(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``-c`` inline exec resolves imports from a -I root."""
        lib_root = tmp_path / "inlinelibs"
        lib_root.mkdir()
        (lib_root / "util.agl").write_text('def greet() -> text = "Hello!"\n')

        # cwd is irrelevant — helper is not reachable from cwd; only via -I
        entry_dir = tmp_path / "work"
        entry_dir.mkdir()

        from agm.config import context as ctx_mod

        original_ctx = ctx_mod.current_config_context()

        class FakeCtx:
            home = original_ctx.home
            proj_dir = original_ctx.proj_dir
            cwd = entry_dir

        monkeypatch.setattr(exec_engine, "current_config_context", lambda: FakeCtx())

        args = ExecArgs(
            file=None,
            command="open import util\nlet r = greet()\nprint r\n",
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=True,
            log_file=None,
            log=False,
            module_paths=[str(lib_root)],
        )
        exec_command.run(args)
        captured = capsys.readouterr()
        assert "Hello!" in captured.out


class TestEntryModuleConfig:
    """A file entry uses its stem as the qualified config module component."""

    def test_entry_stem_selects_root_param_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text("[foo]\nlimit = 7\n")
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        agl_file = tmp_path / "foo.agl"
        write_file_program(agl_file, "param limit: int\nprogram def main() -> unit = print limit\n")

        assert exec_command.run(_exec_args_no_log(agl_file)) is None
        assert capsys.readouterr().out == "7\n"

    def test_qualified_engine_config_conflict_exits_cleanly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from agm.commands import exec_program
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[main.main]\nmax-iters = 1\n\n["tools/main".main]\nmax-iters = 1\n'
        )
        agl_file = tmp_path / "main.agl"
        write_file_program(agl_file, 'program def main() -> unit = print "unreached"\n')

        monkeypatch.setattr(
            exec_program,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_program.run(_exec_args_no_log(agl_file), entry_module_segments=("tools", "main"))

        assert exc_info.value.code == 1
        assert "Error: invalid exec configuration" in capsys.readouterr().err

    def test_entry_module_config_selects_param_despite_imported_name_collision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A bare file-stem key belongs to the entry module, not an import."""
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[workflow]\nregion = "configured"\n')
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        (tmp_path / "settings.agl").write_text('param region: text = "imported"\n')
        agl_file = tmp_path / "workflow.agl"
        write_file_program(
            agl_file,
            "import settings\nparam region: text\nprogram def main() -> unit = print region\n",
        )

        assert exec_command.run(_exec_args_no_log(agl_file)) is None
        assert capsys.readouterr().out == "configured\n"

    def test_engine_key_named_param_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        agl_file = tmp_path / "workflow.agl"
        write_file_program(agl_file, "param timeout: text\nprogram def main() -> unit = ()\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))

        assert exc_info.value.code == 1

    def test_reserved_entry_stem_cannot_declare_params(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "exec.agl"
        write_file_program(agl_file, 'param region: text = "eu"\n')

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))

        assert exc_info.value.code == 1

    def test_reserved_entry_stem_without_params_runs_normally(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "exec.agl"
        agl_file.write_text('program def main() -> unit = print "usable"\n')

        assert exec_command.run(_exec_args_no_log(agl_file)) is None
        assert capsys.readouterr().out == "usable\n"

    def test_reserved_entry_stem_still_selects_among_several_programs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The reserved-stem scan must not consume the program declarations."""
        agl_file = tmp_path / "exec.agl"
        agl_file.write_text(
            'program def first() -> unit = print "first"\n'
            'program def second() -> unit = print "second"\n'
        )

        assert exec_command.run(_exec_args_no_log(agl_file, program="second")) is None
        assert capsys.readouterr().out == "second\n"

    def test_module_table_engine_settings_and_subtables_do_not_warn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Params, engine settings, and nested route tables are all legitimate."""
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            "[workflow]\n"
            "retries = 3\n"
            'timeout = "30s"\n'
            "\n[workflow.main]\n"
            "max-iters = 5\n"
            '\n[workflow.Deploy]\nregion = "prod"\n'
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        agl_file = tmp_path / "workflow.agl"
        agl_file.write_text(
            "param retries: int = 1\n"
            "scope Deploy\n"
            'param region: text = "eu"\n'
            "end Deploy\n"
            "program def main() -> unit = print(Deploy::region)\n"
        )

        assert exec_command.run(_exec_args_no_log(agl_file)) is None
        captured = capsys.readouterr()
        assert captured.out == "prod\n"
        assert captured.err == ""

    def test_key_claimed_by_an_imported_module_suffix_route_does_not_warn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An imported route that reads the entry table by suffix claims its key."""
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text("[demo]\nretries = 5\n")
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "demo.agl").write_text(
            "param retries: int = 1\ndef read() -> int = retries\n"
        )
        agl_file = tmp_path / "demo.agl"
        agl_file.write_text(
            "import lib/demo\nprogram def main() -> unit = print(lib/demo::read())\n"
        )

        assert exec_command.run(_exec_args_no_log(agl_file)) is None
        captured = capsys.readouterr()
        assert captured.out == "5\n"
        assert captured.err == ""

    def test_undeclared_key_beside_valid_keys_warns_once_and_runs_on_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A misspelled param key is reported instead of being silently dropped."""
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[workflow]\nretires = 5\nregion = "prod"\n\n[workflow.main]\nmax-iters = 5\n'
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        agl_file = tmp_path / "workflow.agl"
        agl_file.write_text(
            "param retries: int = 1\n"
            'param region: text = "eu"\n'
            "program def main() -> unit = print retries\n"
        )

        assert exec_command.run(_exec_args_no_log(agl_file)) is None
        captured = capsys.readouterr()
        assert captured.out == "1\n"
        reported = [line for line in captured.err.splitlines() if line.strip()]
        assert len(reported) == 1
        assert "retires" in reported[0]

    def test_cli_max_iters_overrides_selected_qualified_program_table(
        self,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A CLI engine flag wins over the table for the ``-p``-selected program."""
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            "[workflow.first]\nmax-iters = 3\n\n[workflow.second]\nmax-iters = 1\n"
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        agl_file = tmp_path / "workflow.agl"
        agl_file.write_text(
            'program def first() -> unit = print "first"\n'
            "program def second() -> unit =\n"
            "  var count = 0\n"
            "  do\n"
            "    count := count + 1\n"
            "  until count >= 2\n"
            '  print "second"\n'
        )

        configured = invoke(runner, ["exec", "-p", "second", str(agl_file)])
        assert configured.exit_code == 2

        overridden = invoke(
            runner,
            ["exec", "--max-iters", "2", "-p", "second", str(agl_file)],
        )
        assert overridden.exit_code == 0
        assert overridden.output == "second\n"

    def test_qualified_program_table_supplies_multiple_engine_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            "[workflow.main]\nstrict-json = true\nmax-iters = 2\n"
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        agl_file = tmp_path / "workflow.agl"
        agl_file.write_text('program def main() -> unit = print "configured"\n')

        assert exec_command.run(_exec_args_no_log(agl_file)) is None
        assert capsys.readouterr().out == "configured\n"

    def test_imported_params_resolve_by_qualified_config_and_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[settings]\nregion = "configured"\n')
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        (tmp_path / "settings.agl").write_text("param region: text\ndef read() -> text = region\n")
        agl_file = tmp_path / "workflow.agl"
        write_file_program(
            agl_file,
            "import settings\nprogram def main() -> unit = print(settings::read())\n",
        )

        assert exec_command.run(_exec_args_no_log(agl_file)) is None
        assert capsys.readouterr().out == "configured\n"
        assert (
            exec_command.run(
                _exec_args_no_log(agl_file, param_tokens=["--settings::region", "flag"])
            )
            is None
        )
        assert capsys.readouterr().out == "flag\n"

    def test_config_value_for_a_reserved_flag_colliding_param_is_not_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A param named like a built-in exec flag (e.g. ``dry-run``, reserved
        even though it is not an engine setting) is ambiguous on the CLI, so
        its config-table value must be addressed by the SAME qualified
        spelling used by the CLI flag rule — not silently stored under the
        plain name where the runtime would never find it.
        """
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[settings]\ndry-run = "configured"\n')
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        (tmp_path / "settings.agl").write_text(
            "param dry-run: text\ndef read() -> text = dry-run\n"
        )
        agl_file = tmp_path / "workflow.agl"
        write_file_program(
            agl_file,
            "import settings\nprogram def main() -> unit = print(settings::read())\n",
        )

        assert exec_command.run(_exec_args_no_log(agl_file)) is None
        assert capsys.readouterr().out == "configured\n"

    def test_qualified_module_table_supplies_multiple_params(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[settings]\nregion = "configured"\nzone = "secondary"\n'
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        (tmp_path / "settings.agl").write_text(
            'param region: text\nparam zone: text\ndef read() -> text = region + ":" + zone\n'
        )
        agl_file = tmp_path / "workflow.agl"
        write_file_program(
            agl_file,
            "import settings\nprogram def main() -> unit = print(settings::read())\n",
        )

        assert exec_command.run(_exec_args_no_log(agl_file)) is None
        assert capsys.readouterr().out == "configured:secondary\n"

    def test_conflicting_qualified_param_tables_fail_before_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '[settings]\nregion = "short"\n\n["nested/settings"]\nregion = "exact"\n'
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "settings.agl").write_text("param region: text\ndef read() -> text = region\n")
        agl_file = tmp_path / "workflow.agl"
        write_file_program(
            agl_file,
            "import nested/settings\nprogram def main() -> unit = print(settings::read())\n",
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))

        assert exc_info.value.code == 1

    def test_symlinked_entry_stem_collision_keeps_both_qualified_params_settable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        imported_root = tmp_path / "invocation"
        imported_root.mkdir()
        (imported_root / "settings.agl").write_text(
            "param region: text\ndef read() -> text = region\n"
        )
        target_root = tmp_path / "target"
        target_root.mkdir()
        target = target_root / "settings.agl"
        target.write_text(
            "import settings\nparam region: text\n"
            'program def main() -> unit = print(region + ":" + settings::read())\n'
        )
        entry = imported_root / "workflow.agl"
        entry.symlink_to(target)

        assert (
            exec_command.run(
                _exec_args_no_log(
                    entry,
                    param_tokens=[
                        "--@entry::region",
                        "local",
                        "--settings::region",
                        "remote",
                    ],
                )
            )
            is None
        )
        assert capsys.readouterr().out == "local:remote\n"

    def test_qualified_positive_negative_collision_keeps_required_params_settable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "no-settings.agl").write_text(
            "param region: text\ndef read() -> text = region\n"
        )
        (tmp_path / "settings.agl").write_text("param region: bool\n")
        agl_file = tmp_path / "workflow.agl"
        write_file_program(
            agl_file,
            "import no-settings\nimport settings\n"
            "program def main() -> unit = print(no-settings::read())\n",
        )

        assert (
            exec_command.run(
                _exec_args_no_log(
                    agl_file,
                    param_tokens=[
                        "--@module::no-settings::region",
                        "local",
                        "--no-@module::settings::region",
                    ],
                )
            )
            is None
        )
        assert capsys.readouterr().out == "local\n"

    def test_colliding_imported_params_require_qualified_flags(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "one.agl").write_text("param region: text\ndef read() -> text = region\n")
        (tmp_path / "two.agl").write_text("param region: text\ndef read() -> text = region\n")
        agl_file = tmp_path / "workflow.agl"
        write_file_program(
            agl_file,
            "import one\nimport two\n"
            "program def main() -> unit = print(one::read() + two::read())\n",
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file, param_tokens=["--region", "bad"]))
        assert exc_info.value.code == 1
        assert (
            exec_command.run(
                _exec_args_no_log(
                    agl_file,
                    param_tokens=["--one::region", "a", "--two::region", "b"],
                )
            )
            is None
        )
        assert capsys.readouterr().out == "ab\n"


class TestNegatedConstantDefaults:
    """A unary operator over a constant operand is itself a constant.

    Both places that require a constant initializer accept it: an engine
    setting's declared default in ``std/config`` and a module root binding.
    """

    def test_engine_setting_default_may_be_negated(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A substitute standard library negates its ``strict-json`` default."""
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        repo_stdlib = Path(__file__).resolve().parent.parent / "stdlib"
        lib_root = tmp_path / "lib"
        shutil.copytree(repo_stdlib, lib_root)
        config = lib_root / "std" / "config.agl"
        config.write_text(
            config.read_text(encoding="utf-8").replace(
                "builtin var strict-json: bool = false",
                "builtin var strict-json: bool = not false",
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("AGM_STDLIB", str(lib_root))

        entry = tmp_path / "settings.agl"
        entry.write_text(
            "import std/config\nprogram def main() -> unit = print std/config::strict-json\n",
            encoding="utf-8",
        )

        assert exec_command.run(_exec_args_no_log(entry)) is None
        assert capsys.readouterr().out == "true\n"

    def test_root_binding_may_be_negated(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        entry = tmp_path / "threshold.agl"
        entry.write_text(
            "let threshold = -1\n"
            "let disabled = not true\n"
            'program def main() -> unit = print "%{threshold}/%{disabled}"\n',
            encoding="utf-8",
        )

        assert exec_command.run(_exec_args_no_log(entry)) is None
        assert capsys.readouterr().out == "-1/false\n"


class TestSettingOverrideProvenanceWithNoStdlib:
    """``--no-stdlib`` interacts differently with a CLI flag vs. ambient config.

    A ``default-agent`` override reaching the engine as AgL literal source
    (``--agent`` or ``[exec]``/qualified program-table ``default-agent``) can only be
    spliced into ``std/config``'s own declaration when that module is loaded.
    ``--no-stdlib`` on a program that never explicitly imports ``std/config``
    means it never is. An ambient config value is then simply inert (the key
    does not apply to this run); an explicit ``--agent`` flag is a request the
    host cannot silently drop, so it still fails the run.
    """

    def test_config_default_agent_is_inert_without_stdlib(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A project-configured ``[exec] default-agent`` must not break every
        ``--no-stdlib`` program, including one that never mentions an agent."""
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            "[exec]\ndefault-agent = 'AgentCommand(\"echo cfg\")'\n"
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        agl_file = tmp_path / "plain.agl"
        write_file_program(agl_file, "let x = 1\nprogram def main() -> unit = ()\n")

        result = exec_command.run(_exec_args_no_log(agl_file, no_stdlib=True))
        assert result is None

    def test_agent_flag_still_fails_without_stdlib(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An explicit ``--agent`` literal is a request, not ambient config: it
        must still fail (rather than be silently dropped) when the program
        never loads ``std/config``."""
        agl_file = tmp_path / "plain.agl"
        write_file_program(agl_file, "let x = 1\nprogram def main() -> unit = ()\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file, no_stdlib=True, agent="("))

        assert exc_info.value.code == 1
        assert "--agent" in capsys.readouterr().err


class TestExecProgramSelection:
    """Program-def entry selection requires file programs and wraps inline source."""

    def test_runs_the_sole_program_implicitly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "sole.agl"
        write_file_program(agl_file, 'program def main() -> unit = print "sole"\n')

        assert exec_command.run(_exec_args_no_log(agl_file)) is None
        assert capsys.readouterr().out == "sole\n"

    def test_requires_a_program_when_the_entry_declares_several(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "several.agl"
        write_file_program(
            agl_file,
            'program def first() -> unit = print "first"\n'
            "scope review\n"
            'program def main() -> unit = print "review"\n'
            "end review\n",
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "first" in captured.err
        assert "review::main" in captured.err
        assert captured.out == ""

    def test_rejects_an_unknown_program_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "sole.agl"
        write_file_program(agl_file, 'program def main() -> unit = print "sole"\n')

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file, program="missing"))

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "main" in captured.err
        assert captured.out == ""

    def test_rejects_a_file_without_a_program_definition(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "static-only.agl"
        agl_file.write_text("let value = 1\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))

        assert exc_info.value.code == 1
        assert capsys.readouterr().out == ""

    def test_runs_inline_statements_by_wrapping_them_in_a_program(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = ExecArgs(
            file=None,
            command='let value = "inline"\nprint value\n',
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=True,
            log_file=None,
            log=False,
        )

        assert exec_command.run(args) is None
        assert capsys.readouterr().out == "inline\n"

    def test_inline_execution_without_a_selected_program_runs_only_initializers(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A prepared inline entry with no discovered program does not invoke its main."""
        from agm.agl.pipeline import PipelineDriver as RealRuntime

        class NoProgramDiscoveryRuntime(RealRuntime):
            def discover_params(self, *args: object, **kwargs: object):
                return replace(super().discover_params(*args, **kwargs), programs=())

        monkeypatch.setattr(exec_engine, "PipelineDriver", NoProgramDiscoveryRuntime)
        args = ExecArgs(
            file=None,
            command='print "only selected mains run"',
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=True,
            log_file=None,
            log=False,
        )

        assert exec_command.run(args) is None
        assert capsys.readouterr().out == ""

    def test_selected_program_uses_and_restores_the_pinned_decimal_context(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "decimal.agl"
        write_file_program(agl_file, "program def main() -> unit = print(1.0 / 3.0)\n")
        previous = decimal.getcontext().copy()
        decimal.getcontext().prec = 4
        try:
            assert exec_command.run(_exec_args_no_log(agl_file)) is None
            assert capsys.readouterr().out == "0.3333333333333333333333333333\n"
            assert decimal.getcontext().prec == 4
        finally:
            decimal.setcontext(previous)

    def test_selected_program_deep_recursion_uses_the_agl_call_limit(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "recursive.agl"
        write_file_program(
            agl_file,
            "def recurse(n: int) -> unit = recurse(n + 1)\n"
            "program def main() -> unit = recurse(0)\n",
        )

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file, max_call_depth=512))

        assert exc_info.value.code == 2
        assert "RecursionError" in capsys.readouterr().err

    def test_selected_program_error_has_its_source_location(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = tmp_path / "failure.agl"
        write_file_program(agl_file, "program def main() -> unit = print(1 / 0)\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(agl_file))

        assert exc_info.value.code == 2
        assert "at line 1" in capsys.readouterr().err


class TestProgramLogFilePathResolution:
    """Qualified program log-file paths are anchored to their config directory."""

    def test_program_log_file_relative_resolved_to_config_dir(self, tmp_path: Path) -> None:
        from agm.config.general import load_general_config
        from agm.config.qualified_keys import QualifiedConfigKey, resolve_qualified_values

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[myprog.main]\nlog-file = "my.log"\n')

        config = load_general_config(home=home, proj_dir=None, cwd=tmp_path)
        key = QualifiedConfigKey(("myprog",), ("main",), "log-file")
        log_file_val = resolve_qualified_values(config, (key,))[key]

        assert isinstance(log_file_val, str)
        assert Path(log_file_val).is_absolute()
        assert log_file_val.endswith("my.log")


class TestExecDevelopmentPackages:
    def test_direct_exec_uses_the_package_qualified_config_identity(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from agm.config.context import ConfigContext

        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            '["alpha/main"]\nmessage = "package-qualified"\n'
        )
        monkeypatch.setattr(
            exec_engine,
            "current_config_context",
            lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
        )

        alpha = tmp_path / "alpha"
        (alpha / "alpha").mkdir(parents=True)
        (alpha / "package.toml").write_text('[package]\nname = "alpha"\nversion = "1.0.0"\n')
        entry = alpha / "alpha" / "main.agl"
        entry.write_text(
            'param message: text = "default"\nprogram def main() -> unit = print message\n'
        )

        assert exec_command.run(_exec_args_no_log(entry)) is None
        assert capsys.readouterr().out == "package-qualified\n"

    def test_direct_exec_does_not_mount_package_entry_parent_as_loose_root(
        self, tmp_path: Path
    ) -> None:
        alpha = tmp_path / "alpha"
        nested = alpha / "alpha" / "alpha"
        nested.mkdir(parents=True)
        (alpha / "package.toml").write_text('[package]\nname = "alpha"\nversion = "1.0.0"\n')
        (alpha / "alpha" / "settings.agl").write_text("def answer() -> int = 42\n")
        (nested / "settings.agl").write_text("def wrong() -> int = 0\n")
        entry = alpha / "alpha" / "main.agl"
        entry.write_text(
            "import alpha/settings\n"
            "program def main() -> unit =\n"
            "  let _ = alpha/settings::answer()\n"
        )

        assert exec_command.run(_exec_args_no_log(entry, no_stdlib=True)) is None

    def test_direct_exec_reports_an_invalid_development_manifest(self, tmp_path: Path) -> None:
        alpha = tmp_path / "alpha"
        (alpha / "alpha").mkdir(parents=True)
        (alpha / "package.toml").write_text('[package]\nname = "alpha"\n')
        entry = alpha / "alpha" / "main.agl"
        entry.write_text("program def main() -> unit = ()\n")

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(_exec_args_no_log(entry))

        assert exc_info.value.code == 1

    def test_exec_mounts_the_containing_package_and_its_path_dependencies(
        self, tmp_path: Path
    ) -> None:
        bravo = tmp_path / "bravo"
        (bravo / "bravo").mkdir(parents=True)
        (bravo / "package.toml").write_text('[package]\nname = "bravo"\nversion = "1.0.0"\n')
        (bravo / "bravo" / "shared.agl").write_text("def answer() -> int = 42\n")

        alpha = tmp_path / "alpha"
        (alpha / "alpha").mkdir(parents=True)
        (alpha / "package.toml").write_text(
            '[package]\nname = "alpha"\nversion = "1.0.0"\n\n'
            "[dependencies]\n"
            'bravo = { version = "1", path = "../bravo" }\n'
        )
        entry = alpha / "alpha" / "main.agl"
        entry.write_text(
            "import bravo/shared\nprogram def main() -> unit =\n  let _ = bravo/shared::answer()\n"
        )

        assert exec_command.run(_exec_args_no_log(entry, no_stdlib=True)) is None
