"""Acceptance tests for value-driven AgL Agent dispatch."""

from __future__ import annotations

import pytest

from agm.agl import PipelineDriver
from agm.agl.runtime.agents import value_driven_agent_factory
from agm.agl.semantics.values import EnumValue
from tests._agl_helpers import agent_value
from tests.conftest import FakeAgentTransport


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
            ["codex", "exec", "--model", "o3", "-c", "model_reasoning_effort=high", "-"],
        ),
        (
            'AgentPi("openai", "gpt", "low")',
            ["pi", "-p", "--provider", "openai", "--model", "gpt", "--thinking", "low"],
        ),
    ],
)
def test_ask_dispatches_each_agent_value(
    fake_agent_transport: FakeAgentTransport, source: str, argv: list[str]
) -> None:
    fake_agent_transport.queue(fake_agent_transport.success("ok"))
    runtime = PipelineDriver(agent_dispatcher=value_driven_agent_factory(idle_timeout=None))
    result = runtime.run(f'let answer: text = ask("hello", agent = {source})\nanswer')

    assert result.ok
    assert fake_agent_transport.calls == [("hello", argv)]


@pytest.mark.parametrize("failure", ["spawn_error", "timed_out", "returncode"])
def test_agent_transport_failures_become_typed_errors(
    fake_agent_transport: FakeAgentTransport, failure: str
) -> None:
    fake_agent_transport.queue(
        fake_agent_transport.failure(
            spawn_error="failed" if failure == "spawn_error" else None,
            timed_out=failure == "timed_out",
            returncode=2 if failure == "returncode" else None,
            stderr="error output",
            elapsed=1.0,
        )
    )
    runtime = PipelineDriver(agent_dispatcher=value_driven_agent_factory(idle_timeout=None))

    run = runtime.run('let answer: text = ask("hello", agent = AgentCommand("runner"))\nanswer')

    assert not run.ok
    assert run.error is not None
    assert run.error.type_name == "AgentCallError"


def test_codex_agent_dispatch_delivers_prompt_via_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The full value-driven dispatch path sends codex's prompt as stdin, not ``@<path>``."""
    from agm.agl.runtime.request import AgentRequest
    from agm.core.process import ProcessCaptureResult

    captured: dict[str, object] = {}

    def fake_run_capture_result(cmd: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured["cmd"] = cmd
        captured["stdin_text"] = kwargs.get("stdin_text")
        return ProcessCaptureResult(
            returncode=0,
            stdout="ok",
            stderr="",
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
            spawn_errno=None,
        )

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)

    agent = agent_value("AgentCodex", model="o3", thinking="high")
    dispatch = value_driven_agent_factory(idle_timeout=None)

    response = dispatch(AgentRequest(agent=agent, prompt="hello"))

    assert response.content == "ok"
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert not any(str(element).startswith("@") for element in cmd)
    assert cmd[-1] == "-"
    assert captured["stdin_text"] == "hello"


def test_unrecognized_agent_variant_becomes_a_typed_error() -> None:
    """An Agent variant with no host builder cannot leak an untyped host failure."""
    from agm.agl.runtime.agents import AgentCallHostError
    from agm.agl.runtime.request import AgentRequest

    unknown = agent_value("AgentFuture")
    dispatch = value_driven_agent_factory(idle_timeout=None)

    with pytest.raises(AgentCallHostError) as exc_info:
        dispatch(AgentRequest(agent=unknown, prompt="hello"))

    assert exc_info.value.cause == "invalid_agent"


def test_unresolvable_command_hole_becomes_a_typed_error() -> None:
    """A host interpolation hole the environment cannot fill fails as a spawn failure."""
    from agm.agl.runtime.agents import AgentCallHostError
    from agm.agl.runtime.request import AgentRequest

    agent = agent_value("AgentCommand", command="runner --flag=%{AGM_NO_SUCH_VARIABLE}")
    dispatch = value_driven_agent_factory(idle_timeout=None)

    with pytest.raises(AgentCallHostError) as exc_info:
        dispatch(AgentRequest(agent=agent, prompt="hello"))

    assert exc_info.value.cause == "spawn_failure"


def test_escaped_command_hole_reaches_the_host_interpolator() -> None:
    """`\\%{` in AgL source passes the hole through for the host to resolve."""
    runtime = PipelineDriver(agent_dispatcher=value_driven_agent_factory(idle_timeout=None))

    run = runtime.run(
        'let answer: text = ask("hello", '
        'agent = AgentCommand("runner --flag=\\%{AGM_NO_SUCH_VARIABLE}"))\nanswer'
    )

    assert not run.ok
    assert run.error is not None
    assert run.error.type_name == "AgentCallError"
    assert run.error.fields["cause"] == "spawn_failure"


def test_invalid_agent_value_becomes_typed_error() -> None:
    runtime = PipelineDriver(agent_dispatcher=value_driven_agent_factory(idle_timeout=None))

    run = runtime.run('let answer: text = ask("hello", agent = AgentCommand(""))\nanswer')

    assert not run.ok
    assert run.error is not None
    assert run.error.type_name == "AgentCallError"


def test_default_agent_value_is_read_at_each_call_and_errors_stay_typed() -> None:
    requests: list[EnumValue] = []

    def agent(request: object) -> str:
        value = getattr(request, "agent")
        assert isinstance(value, EnumValue)
        requests.append(value)
        return "not an integer"

    runtime = PipelineDriver(agent_dispatcher=agent)
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
