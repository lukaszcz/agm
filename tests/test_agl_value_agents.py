"""Acceptance tests for value-driven AgL Agent dispatch."""

from __future__ import annotations

import pytest

from agm.agl import PipelineDriver
from agm.agl.runtime.agents import value_driven_agent_factory
from agm.agl.semantics.values import EnumValue


@pytest.mark.parametrize(
    ("source", "argv"),
    [
        ('AgentCommand("runner --flag")', ["runner", "--flag"]),
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
def test_ask_dispatches_each_agent_value(
    monkeypatch: pytest.MonkeyPatch, source: str, argv: list[str]
) -> None:
    captured: list[list[str]] = []

    def prepare(prompt: str, *, runner: list[str], **_: object) -> object:
        assert prompt == "hello"
        captured.append(runner)
        return object()

    monkeypatch.setattr("agm.agent.runner.prepare_rendered_prompt_run", prepare)
    transport_result = type(
        "Result",
        (),
        {
            "spawn_error": None,
            "timed_out": False,
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "elapsed": 0.0,
        },
    )()
    monkeypatch.setattr(
        "agm.agent.runner.run_prepared_prompt_result",
        lambda _prepared, **_: transport_result,
    )
    runtime = PipelineDriver(value_agent=value_driven_agent_factory(idle_timeout=None))
    result = runtime.run(f'let answer: text = ask("hello", agent = {source})\nanswer')

    assert result.ok
    assert captured == [argv]


def test_default_agent_value_is_read_at_each_call_and_errors_stay_typed() -> None:
    requests: list[EnumValue] = []

    def agent(request: object) -> str:
        value = getattr(request, "agent")
        assert isinstance(value, EnumValue)
        requests.append(value)
        return "not an integer"

    runtime = PipelineDriver(value_agent=agent)
    result = runtime.run(
        "import std/config\n"
        'std/config::default-agent := AgentCommand("first")\n'
        'let first: text = ask("one")\n'
        'std/config::default-agent := AgentClaude("sonnet", "medium")\n'
        'let second: int = ask("two", on_parse_error = Retry(n = 0))\n'
        "second"
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "AgentParseError"
    assert result.error.fields["agent"] == {
        "$case": "AgentClaude",
        "model": "sonnet",
        "thinking": "medium",
    }
    assert [request.variant for request in requests] == ["AgentCommand", "AgentClaude"]
