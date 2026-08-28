"""End-to-end ``agm exec`` acceptance tests for value-driven agents."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from typer.main import get_command

import agm.cli as cli
from agm.commands import exec_program as exec_engine
from agm.config.context import ConfigContext
from tests._agl_helpers import write_file_program
from tests.conftest import FakeAgentTransport

# Building the Typer app's Click command tree costs more than the assertions in
# some of these tests; it is derived from module-level definitions only, so it is
# built once here instead of on every invocation.
_AGM_COMMAND = get_command(cli.app)


def _invoke(runner: CliRunner, argv: list[str]):
    return runner.invoke(_AGM_COMMAND, argv, prog_name="agm", catch_exceptions=False)


def test_exec_agent_method_single_attempt_accepts_command_without_session_id(
    tmp_path: Path, fake_agent_transport: FakeAgentTransport
) -> None:
    program = tmp_path / "program.agl"
    write_file_program(
        program,
        'let answer: text = AgentCommand("command --flag").ask("hello")\nprint answer\n',
    )
    fake_agent_transport.queue(fake_agent_transport.success("done"))

    result = _invoke(CliRunner(), ["exec", "--no-log", str(program)])

    assert result.exit_code == 0, result.output
    assert result.output == "done\n"
    prompt, argv = fake_agent_transport.calls[0]
    assert prompt == "hello"
    assert argv[:2] == ["command", "--flag"]


def test_exec_agent_method_command_session_substitutes_its_session_id(
    tmp_path: Path, fake_agent_transport: FakeAgentTransport
) -> None:
    program = tmp_path / "program.agl"
    write_file_program(
        program,
        'let answer: text = AgentCommand("command --flag \\%{SESSION_ID}").ask("hello")\n'
        "print answer\n",
    )
    fake_agent_transport.queue(fake_agent_transport.success("done"))

    result = _invoke(CliRunner(), ["exec", "--no-log", str(program)])

    assert result.exit_code == 0, result.output
    assert result.output == "done\n"
    prompt, argv = fake_agent_transport.calls[0]
    assert prompt == "hello"
    assert argv[:2] == ["command", "--flag"]
    assert len(argv) == 3


def test_exec_runner_default_session_uses_a_session_id_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_agent_transport: FakeAgentTransport,
) -> None:
    home = tmp_path / "home"
    config_dir = home / ".agm"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('[exec]\nrunner = "command %{SESSION_ID}"\n')
    program = tmp_path / "program.agl"
    write_file_program(program, 'let answer: text = ask("hello")\nprint answer\n')
    monkeypatch.setattr(
        exec_engine,
        "current_config_context",
        lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
    )
    fake_agent_transport.queue(fake_agent_transport.success("done"))

    result = _invoke(CliRunner(), ["exec", "--no-log", str(program)])

    assert result.exit_code == 0, result.output
    assert result.output == "done\n"
    prompt, argv = fake_agent_transport.calls[0]
    assert prompt == "hello"
    assert argv[0] == "command"
    assert len(argv) == 2


def test_exec_runner_without_a_session_id_placeholder_fails_free_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    config_dir = home / ".agm"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('[exec]\nrunner = "command"\n')
    program = tmp_path / "program.agl"
    write_file_program(program, 'let answer: text = ask("hello")\nprint answer\n')
    monkeypatch.setattr(
        exec_engine,
        "current_config_context",
        lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
    )

    result = _invoke(CliRunner(), ["exec", "--no-log", str(program)])

    assert result.exit_code != 0
    assert "%{SESSION_ID}" in result.output
    assert "AgentCommand.ask" in result.output


@pytest.mark.parametrize(
    ("cli_agent", "source_agent", "expected_command"),
    [
        (None, None, "config"),
        ('AgentCommand("cli \\%{SESSION_ID}")', None, "cli"),
        (
            'AgentCommand("cli \\%{SESSION_ID}")',
            'AgentCommand("source \\%{SESSION_ID}")',
            "source",
        ),
    ],
)
def test_exec_default_agent_precedence_is_config_then_cli_then_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_agent_transport: FakeAgentTransport,
    cli_agent: str | None,
    source_agent: str | None,
    expected_command: str,
) -> None:
    home = tmp_path / "home"
    config_dir = home / ".agm"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        "[exec]\ndefault-agent = 'AgentCommand(\"config \\%{SESSION_ID}\")'\n"
    )
    source_write = "" if source_agent is None else f"std/config::default-agent := {source_agent}\n"
    program = tmp_path / "program.agl"
    write_file_program(
        program, f'import std/config\n{source_write}let answer: text = ask("hello")\nprint answer\n'
    )
    monkeypatch.setattr(
        exec_engine,
        "current_config_context",
        lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
    )
    fake_agent_transport.queue(fake_agent_transport.success("done"))
    argv = ["exec", "--no-log"]
    if cli_agent is not None:
        argv.extend(["--agent", cli_agent])
    argv.append(str(program))

    result = _invoke(CliRunner(), argv)

    assert result.exit_code == 0, result.output
    prompt, command = fake_agent_transport.calls[0]
    assert prompt == "hello"
    assert command[0] == expected_command
    assert len(command) == 2


def test_exec_retries_with_the_output_contract_feedback(
    tmp_path: Path, fake_agent_transport: FakeAgentTransport
) -> None:
    program = tmp_path / "program.agl"
    write_file_program(
        program,
        'let answer: int = AgentCommand("mock \\%{SESSION_ID}").ask("count", '
        "on_parse_error = Retry(n = 1))\n"
        "print answer\n",
    )
    fake_agent_transport.queue(
        fake_agent_transport.success("not an integer"), fake_agent_transport.success("7")
    )

    result = _invoke(CliRunner(), ["exec", "--no-log", str(program)])

    assert result.exit_code == 0, result.output
    assert result.output == "7\n"
    assert [argv[0] for _, argv in fake_agent_transport.calls] == ["mock", "mock"]
    assert "validation errors:" in fake_agent_transport.calls[1][0].lower()


@pytest.mark.parametrize(
    ("result", "caught_type"),
    [
        (FakeAgentTransport.failure(spawn_error="missing executable"), "AgentCallError"),
        (FakeAgentTransport.success("not an integer"), "AgentParseError"),
    ],
)
def test_exec_typed_agent_errors_retain_the_selected_agent_value(
    tmp_path: Path,
    fake_agent_transport: FakeAgentTransport,
    result: SimpleNamespace,
    caught_type: str,
) -> None:
    program = tmp_path / "program.agl"
    write_file_program(
        program,
        "try\n"
        '  let answer: int = AgentCommand("mock \\%{SESSION_ID}").ask("count")\n'
        "  print answer\n"
        f"catch {caught_type} as error =>\n"
        "  print render(error.agent)\n",
    )
    fake_agent_transport.queue(result)

    invocation = _invoke(CliRunner(), ["exec", "--no-log", str(program)])

    assert invocation.exit_code == 0, invocation.output
    assert "AgentCommand" in invocation.output
    assert "mock" in invocation.output
