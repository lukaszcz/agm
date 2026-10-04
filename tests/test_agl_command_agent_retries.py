"""``AgentCommand.ask`` parse retries through the real session host, with a mocked process."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agent.session import AglSessionHost, SessionService
from agm.agent.session.cli_adapters import open_cli_session
from agm.agl import PipelineDriver
from agm.agl.pipeline import RunResult
from agm.core.process import CapturedOutput, ProcessCaptureResult
from tests._agl_helpers import agl_roots, run_inline_code, unavailable_sandbox_context


class _Command:
    """Replays scripted outputs and records every prompt the command was invoked with."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.prompts: list[str] = []

    def __call__(self, argv: list[str], **_kwargs: object) -> ProcessCaptureResult:
        prompt_files = [arg[1:] for arg in argv if arg.startswith("@")]
        self.prompts.append(Path(prompt_files[0]).read_text(encoding="utf-8"))
        output = self.outputs[len(self.prompts) - 1]
        return ProcessCaptureResult(
            returncode=0,
            stdout=CapturedOutput(data=output.encode(), truncated=False),
            stderr=CapturedOutput(data=b"", truncated=False),
            elapsed=0.0,
            timed_out=False,
            spawn_error=None,
        )


def _run(monkeypatch: pytest.MonkeyPatch, command: _Command, source: str) -> RunResult:
    monkeypatch.setattr("agm.agent.runner.run_capture_result", command)
    host = AglSessionHost(
        SessionService(
            lambda request: open_cli_session(
                request, idle_timeout=None, get_sandbox_context=unavailable_sandbox_context
            )
        )
    )
    driver = PipelineDriver(resolve_agent_spec=None, get_sandbox_context=None, session_host=host)
    return run_inline_code(driver, source, roots=agl_roots())


def _ask(command: str, target: str, retries: int) -> str:
    return (
        "program def main() -> unit =\n"
        f'  let answer = AgentCommand("{command}").ask::[{target}]("question", '
        f"sandbox = AgentSandbox::Disabled, parse-error-retries = {retries})\n"
        "  print(answer)\n"
    )


def test_command_without_placeholder_retries_with_complete_one_shot_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _Command(["not a number", "still not", "42"])

    result = _run(monkeypatch, command, _ask("runner", "int", 2))

    assert result.ok
    assert len(command.prompts) == 3
    assert all("question" in prompt for prompt in command.prompts)
    assert "not a number" in command.prompts[1]
    assert "still not" in command.prompts[2]


def test_command_without_placeholder_exhausts_its_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _Command(["bad", "worse"])

    result = _run(monkeypatch, command, _ask("runner", "int", 1))

    assert result.error is not None
    assert result.error.type_name == "AgentParseError"
    assert len(command.prompts) == 2


def test_command_with_placeholder_retries_as_short_follow_ups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _Command(["not a number", "42"])

    result = _run(monkeypatch, command, _ask("runner --session \\%{SESSION_ID}", "int", 1))

    assert result.ok
    assert "question" not in command.prompts[1]
    assert "not a number" not in command.prompts[1]


def test_text_target_with_retries_needs_no_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _Command(["hello"])

    result = _run(monkeypatch, command, _ask("runner", "text", 2))

    assert result.ok
    assert len(command.prompts) == 1


def test_explicit_session_without_placeholder_still_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _Command(["hello"])
    source = (
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentCommand("runner"))\n'
        '  print(session.ask::[text]("hi"))\n'
    )

    result = _run(monkeypatch, command, source)

    assert result.error is not None
    assert command.prompts == []
