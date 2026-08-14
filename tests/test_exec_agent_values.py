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


def _invoke(runner: CliRunner, argv: list[str]):
    return runner.invoke(get_command(cli.app), argv, prog_name="agm", catch_exceptions=False)


@pytest.mark.parametrize(
    ("agent", "expected_argv"),
    [
        ('AgentCommand("command --flag")', ["command", "--flag"]),
        (
            'AgentClaude("sonnet", "medium")',
            ["claude", "-p", "--model", "sonnet", "--effort", "medium"],
        ),
        (
            'AgentCodex("o3", "high")',
            ["codex", "exec", "--model", "o3", "-c", "model_reasoning_effort=high", "-"],
        ),
        (
            'AgentPi("openai", "gpt", "low")',
            ["pi", "-p", "--provider", "openai", "--model", "gpt", "--thinking", "low"],
        ),
    ],
)
def test_exec_dispatches_each_agent_value_through_its_builder(
    tmp_path: Path,
    fake_agent_transport: FakeAgentTransport,
    agent: str,
    expected_argv: list[str],
) -> None:
    program = tmp_path / "program.agl"
    write_file_program(program, f'let answer: text = ask("hello", agent = {agent})\nprint answer\n')
    fake_agent_transport.queue(fake_agent_transport.success("done"))

    result = _invoke(CliRunner(), ["exec", "--no-log", str(program)])

    assert result.exit_code == 0, result.output
    assert result.output == "done\n"
    assert fake_agent_transport.calls == [("hello", expected_argv)]


@pytest.mark.parametrize(
    ("cli_agent", "source_agent", "expected_command"),
    [
        (None, None, "config"),
        ('AgentCommand("cli")', None, "cli"),
        ('AgentCommand("cli")', 'AgentCommand("source")', "source"),
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
    (config_dir / "config.toml").write_text("[exec]\ndefault-agent = 'AgentCommand(\"config\")'\n")
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
    assert fake_agent_transport.calls == [("hello", [expected_command])]


def test_exec_retries_with_the_output_contract_feedback(
    tmp_path: Path, fake_agent_transport: FakeAgentTransport
) -> None:
    program = tmp_path / "program.agl"
    write_file_program(
        program,
        'let answer: int = ask("count", agent = AgentCommand("mock"), '
        "on_parse_error = Retry(n = 1))\n"
        "print answer\n",
    )
    fake_agent_transport.queue(
        fake_agent_transport.success("not an integer"), fake_agent_transport.success("7")
    )

    result = _invoke(CliRunner(), ["exec", "--no-log", str(program)])

    assert result.exit_code == 0, result.output
    assert result.output == "7\n"
    assert [argv for _, argv in fake_agent_transport.calls] == [["mock"], ["mock"]]
    assert "previous response did not match" in fake_agent_transport.calls[1][0].lower()


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
        '  let answer: int = ask("count", agent = AgentPi("openai", "gpt", "low"))\n'
        "  print answer\n"
        f"catch {caught_type} as error =>\n"
        "  print render(error.agent)\n",
    )
    fake_agent_transport.queue(result)

    invocation = _invoke(CliRunner(), ["exec", "--no-log", str(program)])

    assert invocation.exit_code == 0, invocation.output
    assert "AgentPi" in invocation.output
    assert "openai" in invocation.output
