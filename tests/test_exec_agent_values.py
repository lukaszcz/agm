"""End-to-end ``agm exec`` acceptance tests for value-driven agents."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from typer.main import get_command

import agm.cli as cli
import agm.commands.exec as exec_command
from agm.config.context import ConfigContext


def _invoke(runner: CliRunner, argv: list[str]):
    return runner.invoke(get_command(cli.app), argv, prog_name="agm", catch_exceptions=False)


def _mock_runner(
    monkeypatch: pytest.MonkeyPatch, responses: list[SimpleNamespace]
) -> list[tuple[str, list[str]]]:
    calls: list[tuple[str, list[str]]] = []

    def prepare(prompt: str, *, runner: list[str], **_: object) -> object:
        calls.append((prompt, runner))
        return object()

    monkeypatch.setattr("agm.agent.runner.prepare_rendered_prompt_run", prepare)
    monkeypatch.setattr(
        "agm.agent.runner.run_prepared_prompt_result",
        lambda _prepared, **_: responses.pop(0),
    )
    return calls


def _success(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        spawn_error=None,
        timed_out=False,
        returncode=0,
        stdout=text,
        stderr="",
        elapsed=0.0,
    )


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
            ["codex", "exec", "--model", "o3", "-c", "model_reasoning_effort=high"],
        ),
        (
            'AgentPi("openai", "gpt", "low")',
            ["pi", "-p", "--provider", "openai", "--model", "gpt", "--thinking", "low"],
        ),
    ],
)
def test_exec_dispatches_each_agent_value_through_its_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
    expected_argv: list[str],
) -> None:
    program = tmp_path / "program.agl"
    program.write_text(f'let answer: text = ask("hello", agent = {agent})\nprint answer\n')
    calls = _mock_runner(monkeypatch, [_success("done")])

    result = _invoke(CliRunner(), ["exec", "--no-log", str(program)])

    assert result.exit_code == 0, result.output
    assert result.output == "done\n"
    assert calls == [("hello", expected_argv)]


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
    program.write_text(
        f'import std/config\n{source_write}let answer: text = ask("hello")\nprint answer\n'
    )
    monkeypatch.setattr(
        exec_command,
        "current_config_context",
        lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
    )
    calls = _mock_runner(monkeypatch, [_success("done")])
    argv = ["exec", "--no-log"]
    if cli_agent is not None:
        argv.extend(["--agent", cli_agent])
    argv.append(str(program))

    result = _invoke(CliRunner(), argv)

    assert result.exit_code == 0, result.output
    assert calls == [("hello", [expected_command])]


def test_exec_retries_with_the_output_contract_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    program = tmp_path / "program.agl"
    program.write_text(
        'let answer: int = ask("count", agent = AgentCommand("mock"), '
        "on_parse_error = Retry(n = 1))\n"
        "print answer\n"
    )
    calls = _mock_runner(monkeypatch, [_success("not an integer"), _success("7")])

    result = _invoke(CliRunner(), ["exec", "--no-log", str(program)])

    assert result.exit_code == 0, result.output
    assert result.output == "7\n"
    assert [argv for _, argv in calls] == [["mock"], ["mock"]]
    assert "previous response did not match" in calls[1][0].lower()


@pytest.mark.parametrize(
    ("result", "caught_type"),
    [
        (
            SimpleNamespace(
                spawn_error="missing executable",
                timed_out=False,
                returncode=None,
                stdout="",
                stderr="",
                elapsed=0.0,
            ),
            "AgentCallError",
        ),
        (_success("not an integer"), "AgentParseError"),
    ],
)
def test_exec_typed_agent_errors_retain_the_selected_agent_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: SimpleNamespace,
    caught_type: str,
) -> None:
    program = tmp_path / "program.agl"
    program.write_text(
        "try\n"
        '  let answer: int = ask("count", agent = AgentPi("openai", "gpt", "low"))\n'
        "  print answer\n"
        f"catch {caught_type} as error =>\n"
        "  print render(error.agent)\n"
    )
    _mock_runner(monkeypatch, [result])

    invocation = _invoke(CliRunner(), ["exec", "--no-log", str(program)])

    assert invocation.exit_code == 0, invocation.output
    assert "AgentPi" in invocation.output
    assert "openai" in invocation.output
