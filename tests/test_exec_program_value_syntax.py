"""End-to-end coverage for AgL value syntax on ``agm exec``'s host value
surfaces: a CLI flag token, an ``@opt-env`` value, a qualified TOML config
string, an ``Option[T]`` flag, host Agent syntax on a plain and an
``Option[Agent]`` surface, and ``json``/``Option[json]``-typed config values
that stay literal JSON data instead of being read as value syntax.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.commands.exec as exec_command
from tests._agl_helpers import write_file_program
from tests.test_exec_command import _config_home, _exec_args_no_log


def _point_program(tmp_path: Path) -> Path:
    agl_file = tmp_path / "prog.agl"
    write_file_program(
        agl_file,
        "record Point\n"
        "  x: int\n"
        "  y: int\n\n"
        "program def main(\n"
        '  @opt-env("POINT")\n'
        "  point: Point,\n"
        "  tag: Option[Point] = Option::None,\n"
        "  meta: json = {},\n"
        ") -> unit =\n"
        '  print "%{point.x},%{point.y}"\n'
        "  print (case tag of\n"
        '    | Option::Some(value) => "tag %{value.x},%{value.y}"\n'
        '    | Option::None => "no tag")\n'
        "  print meta\n",
    )
    return agl_file


def _agent_program(tmp_path: Path) -> Path:
    agl_file = tmp_path / "prog.agl"
    write_file_program(
        agl_file,
        "def describe(agent: Agent) -> text =\n"
        "  case agent of\n"
        '    | AgentClaude(model, thinking) => "claude %{model} %{thinking}"\n'
        '    | AgentCodex(model, thinking) => "codex %{model} %{thinking}"\n'
        '    | AgentPi(provider, model, thinking) => "pi %{provider} %{model} %{thinking}"\n'
        '    | AgentCommand(command) => "command %{command}"\n\n'
        "program def main(\n"
        '  worker: Agent = AgentCommand("noop"),\n'
        "  backup: Option[Agent] = Option::None,\n"
        ") -> unit =\n"
        "  print describe(worker)\n"
        "  print (case backup of\n"
        "    | Option::Some(value) => describe(value)\n"
        '    | Option::None => "no backup")\n',
    )
    return agl_file


def _numeric_program(tmp_path: Path) -> Path:
    agl_file = tmp_path / "prog.agl"
    write_file_program(
        agl_file,
        "program def main(\n"
        "  ratio: Option[decimal] = Option::None,\n"
        "  extra: Option[json] = Option::None,\n"
        ") -> unit =\n"
        "  print (case ratio of\n"
        '    | Option::Some(value) => "ratio %{value}"\n'
        '    | Option::None => "no ratio")\n'
        "  print (case extra of\n"
        '    | Option::Some(value) => "extra %{value}"\n'
        '    | Option::None => "no extra")\n',
    )
    return agl_file


class TestValueSyntaxOnHostSurfaces:
    def test_cli_flag_token_reads_a_constructor_call_as_value_syntax(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = _point_program(tmp_path)

        assert (
            exec_command.run(
                _exec_args_no_log(agl_file, argument_tokens=["--point", "Point(x = 1, y = 2)"])
            )
            is None
        )

        out = capsys.readouterr().out
        assert out.splitlines()[0] == "1,2"

    def test_opt_env_value_reads_value_syntax(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = _point_program(tmp_path)
        monkeypatch.setenv("POINT", "Point(x = 3, y = 4)")

        assert exec_command.run(_exec_args_no_log(agl_file)) is None

        out = capsys.readouterr().out
        assert out.splitlines()[0] == "3,4"

    def test_qualified_toml_config_string_reads_value_syntax(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _config_home(tmp_path, monkeypatch, '[prog.main]\npoint = "Point(x = 5, y = 6)"\n')
        agl_file = _point_program(tmp_path)
        monkeypatch.delenv("POINT", raising=False)

        assert exec_command.run(_exec_args_no_log(agl_file)) is None

        out = capsys.readouterr().out
        assert out.splitlines()[0] == "5,6"

    def test_option_flag_reads_a_value_syntax_token(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An ``Option[Point]`` flag's token is a real value-syntax literal
        (a positional-then-named constructor call), not plain JSON."""
        agl_file = _point_program(tmp_path)

        assert (
            exec_command.run(
                _exec_args_no_log(
                    agl_file,
                    argument_tokens=[
                        "--point",
                        "Point(x = 0, y = 0)",
                        "--tag",
                        "Point(1, y = 2)",
                    ],
                )
            )
            is None
        )

        out = capsys.readouterr().out
        assert out.splitlines()[1] == "tag 1,2"

    def test_option_agent_flag_reads_host_agent_shorthand(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agl_file = _agent_program(tmp_path)

        assert (
            exec_command.run(
                _exec_args_no_log(agl_file, argument_tokens=["--backup", "claude/opus-high"])
            )
            is None
        )

        out = capsys.readouterr().out
        assert out.splitlines()[-1] == "claude opus high"

    def test_agent_typed_toml_string_reads_host_agent_shorthand(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _config_home(tmp_path, monkeypatch, '[prog.main]\nworker = "codex/o3-high"\n')
        agl_file = _agent_program(tmp_path)

        assert exec_command.run(_exec_args_no_log(agl_file)) is None

        out = capsys.readouterr().out
        assert out.splitlines()[0] == "codex o3 high"

    def test_agent_typed_toml_string_reads_a_constructor_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _config_home(
            tmp_path,
            monkeypatch,
            '[prog.main]\nworker = \'AgentClaude(model = "sonnet", thinking = "high")\'\n',
        )
        agl_file = _agent_program(tmp_path)

        assert exec_command.run(_exec_args_no_log(agl_file)) is None

        out = capsys.readouterr().out
        assert out.splitlines()[0] == "claude sonnet high"

    def test_toml_native_float_decodes_for_an_option_decimal_parameter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Regression: a native TOML float boxed for ``Option[decimal]`` must
        still cross the canonical JSON boundary (float -> Decimal), not skip
        it the way a bare ``OptionSome`` native payload once did."""
        _config_home(tmp_path, monkeypatch, "[prog.main]\nratio = 1.5\n")
        agl_file = _numeric_program(tmp_path)

        assert exec_command.run(_exec_args_no_log(agl_file)) is None

        out = capsys.readouterr().out
        assert out.splitlines()[0] == "ratio 1.5"

    def test_option_json_toml_string_stays_literal_json_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _config_home(tmp_path, monkeypatch, '[prog.main]\nextra = "hello"\n')
        agl_file = _numeric_program(tmp_path)

        assert exec_command.run(_exec_args_no_log(agl_file)) is None

        out = capsys.readouterr().out
        assert out.splitlines()[-1] == 'extra "hello"'

    def test_json_typed_config_table_stays_literal_json_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A native TOML table for a ``json``-typed parameter crosses as data,
        not as value syntax: a config-native value is already typed, so an
        already-native dict is JSON data as-is (see ``native_raw_value``)."""
        _config_home(
            tmp_path,
            monkeypatch,
            '[prog.main]\npoint = "Point(x = 0, y = 0)"\n\n[prog.main.meta]\na = 1\nb = 2\n',
        )
        agl_file = _point_program(tmp_path)
        monkeypatch.delenv("POINT", raising=False)

        assert exec_command.run(_exec_args_no_log(agl_file)) is None

        out = capsys.readouterr().out
        assert out.splitlines()[-1] == '{"a": 1, "b": 2}'

    def test_json_typed_config_string_is_literal_string_data_not_value_syntax(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A native TOML *string* for a ``json``-typed parameter is the
        parameter's own JSON string value, never re-read as value syntax or
        JSON source text — only a CLI token (raw, unparsed text) does that."""
        _config_home(
            tmp_path,
            monkeypatch,
            '[prog.main]\npoint = "Point(x = 0, y = 0)"\nmeta = "Foo(x = 1)"\n',
        )
        agl_file = _point_program(tmp_path)
        monkeypatch.delenv("POINT", raising=False)

        assert exec_command.run(_exec_args_no_log(agl_file)) is None

        out = capsys.readouterr().out
        assert out.splitlines()[-1] == '"Foo(x = 1)"'

    def test_json_typed_cli_flag_rejects_value_syntax_constructors(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A ``json``-typed CLI flag's raw token IS read through the host-text
        dispatch (strict JSON, then value syntax restricted to plain JSON
        shapes) — a constructor call is not valid JSON data, so it is a host
        error naming the parameter."""
        agl_file = _point_program(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            exec_command.run(
                _exec_args_no_log(
                    agl_file,
                    argument_tokens=[
                        "--point",
                        "Point(x = 0, y = 0)",
                        "--meta",
                        "Foo(x = 1)",
                    ],
                )
            )

        assert exc_info.value.code == 1
        assert "meta" in capsys.readouterr().err
