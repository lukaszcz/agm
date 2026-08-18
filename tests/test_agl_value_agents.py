"""Acceptance tests for value-driven AgL Agent dispatch."""

from __future__ import annotations

import os
from dataclasses import fields
from typing import get_type_hints

import pytest

from agm.agent.spec import AGENT_SPECS
from agm.agl import PipelineDriver
from agm.agl.runtime.agents import decode_agent_value, value_driven_agent_factory
from agm.agl.semantics.type_table import BUILTIN_PRELUDE_TYPE_DEFS
from agm.agl.semantics.types import TextType
from agm.agl.semantics.values import EnumValue
from tests._agl_helpers import agent_value, run_inline_command
from tests.conftest import FakeAgentTransport


def test_host_specs_match_declared_agent_variants() -> None:
    """The host decoder catalog must track the checked ``Agent`` prelude shape."""
    declared = {
        variant: tuple(name for name, _ in payload)
        for variant, payload in BUILTIN_PRELUDE_TYPE_DEFS["Agent"].variants
    }

    assert set(AGENT_SPECS) == set(declared)
    for variant, spec_cls in AGENT_SPECS.items():
        spec_fields = fields(spec_cls)
        assert tuple(field.name for field in spec_fields) == spec_cls.PAYLOAD_FIELDS
        assert spec_cls.PAYLOAD_FIELDS == declared[variant]
        hints = get_type_hints(spec_cls)
        assert all(hints[field.name] is str for field in spec_fields)
    assert all(
        isinstance(field_type, TextType)
        for _, payload in BUILTIN_PRELUDE_TYPE_DEFS["Agent"].variants
        for _, field_type in payload
    )


def test_decode_accepts_every_declared_agent_variant() -> None:
    for variant, payload in BUILTIN_PRELUDE_TYPE_DEFS["Agent"].variants:
        value = agent_value(variant, **{name: name for name, _ in payload})

        assert isinstance(decode_agent_value(value), AGENT_SPECS[variant])


def test_decode_rejects_declared_variant_without_a_host_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = dict(AGENT_SPECS)
    del catalog["AgentCommand"]
    monkeypatch.setattr("agm.agent.spec.AGENT_SPECS", catalog)

    with pytest.raises(ValueError):
        decode_agent_value(agent_value("AgentCommand", command="runner"))


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
    result = run_inline_command(
        runtime,
        f'let answer: text = ask("hello", agent = {source})\nanswer',
    )

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

    run = run_inline_command(
        runtime,
        'let answer: text = ask("hello", agent = AgentCommand("runner"))\nanswer',
    )

    assert not run.ok
    assert run.error is not None
    assert run.error.type_name == "AgentCallError"


def test_nonzero_exit_message_includes_the_exit_code(
    fake_agent_transport: FakeAgentTransport,
) -> None:
    """A caught ``AgentCallError.message`` still names the exit code (not just metadata)."""
    fake_agent_transport.queue(
        fake_agent_transport.failure(returncode=7, stderr="boom", elapsed=1.0)
    )
    runtime = PipelineDriver(agent_dispatcher=value_driven_agent_factory(idle_timeout=None))

    run = run_inline_command(
        runtime,
        'let answer: text = ask("hello", agent = AgentCommand("runner"))\nanswer',
    )

    assert not run.ok
    assert run.error is not None
    assert run.error.type_name == "AgentCallError"
    assert "(exit 7)" in run.error.fields["message"]


def test_composed_prompt_is_unchanged_when_no_output_contract(
    fake_agent_transport: FakeAgentTransport,
) -> None:
    """No contract, no retry: the prompt sent is exactly the request's prompt."""
    from agm.agl.runtime.request import AgentRequest

    dispatch = value_driven_agent_factory(idle_timeout=None)
    agent = agent_value("AgentCommand", command="runner")

    dispatch(AgentRequest(agent=agent, prompt="Do X."))

    assert fake_agent_transport.calls == [("Do X.", ["runner"])]


def test_composed_prompt_appends_format_instructions_after_the_prompt(
    fake_agent_transport: FakeAgentTransport,
) -> None:
    from agm.agl.runtime.contract import TypelessOutputContract
    from agm.agl.runtime.request import AgentRequest, compose_agent_prompt

    dispatch = value_driven_agent_factory(idle_timeout=None)
    agent = agent_value("AgentCommand", command="runner")
    contract = TypelessOutputContract(
        target_type="int",
        codec_name="json",
        strict_json=True,
        format_instructions="Return only valid JSON matching the schema.",
        json_schema=None,
    )

    request = AgentRequest(agent=agent, prompt="Do X.", output_contract=contract)
    request.prompt = compose_agent_prompt(request)
    dispatch(request)

    prompt = fake_agent_transport.calls[0][0]
    assert "Do X." in prompt
    assert "Return only valid JSON matching the schema." in prompt
    assert prompt.index("Do X.") < prompt.index("Return only valid JSON matching the schema.")


def test_composed_prompt_omits_format_instructions_when_the_contract_has_none(
    fake_agent_transport: FakeAgentTransport,
) -> None:
    """An empty ``format_instructions`` (e.g. the text codec) adds nothing."""
    from agm.agl.runtime.contract import TypelessOutputContract
    from agm.agl.runtime.request import AgentRequest

    dispatch = value_driven_agent_factory(idle_timeout=None)
    agent = agent_value("AgentCommand", command="runner")
    contract = TypelessOutputContract(
        target_type="text",
        codec_name="text",
        strict_json=None,
        format_instructions="",
        json_schema=None,
    )

    dispatch(AgentRequest(agent=agent, prompt="Do X.", output_contract=contract))

    assert fake_agent_transport.calls == [("Do X.", ["runner"])]


def test_composed_prompt_includes_retry_feedback_on_a_retry_attempt(
    fake_agent_transport: FakeAgentTransport,
) -> None:
    """A retry (attempt >= 1) appends the previous output and validation errors."""
    from agm.agl.runtime.request import AgentRequest, ValidationError, compose_agent_prompt

    dispatch = value_driven_agent_factory(idle_timeout=None)
    agent = agent_value("AgentCommand", command="runner")

    request = AgentRequest(
        agent=agent,
        prompt="Do X.",
        attempt=1,
        previous_invalid_output="the-bad-output-xyz",
        validation_errors=[
            ValidationError(category="missing_field", message="missing field 'name'"),
            ValidationError(category="wrong_type", message="type mismatch: expected int"),
        ],
    )
    request.prompt = compose_agent_prompt(request)
    dispatch(request)

    prompt = fake_agent_transport.calls[0][0]
    assert "Your previous response did not match the required output format" in prompt
    assert "- missing field 'name'" in prompt
    assert "- type mismatch: expected int" in prompt
    assert "the-bad-output-xyz" in prompt
    assert "Return only valid JSON matching the schema." in prompt


def test_composed_prompt_has_no_retry_feedback_on_the_first_attempt(
    fake_agent_transport: FakeAgentTransport,
) -> None:
    from agm.agl.runtime.request import AgentRequest

    dispatch = value_driven_agent_factory(idle_timeout=None)
    agent = agent_value("AgentCommand", command="runner")

    dispatch(AgentRequest(agent=agent, prompt="Do X.", attempt=0))

    prompt = fake_agent_transport.calls[0][0]
    assert "Your previous response did not match" not in prompt


def test_composed_prompt_orders_format_instructions_before_retry_feedback(
    fake_agent_transport: FakeAgentTransport,
) -> None:
    """Ordering: prompt, then format_instructions, then the retry-feedback block."""
    from agm.agl.runtime.contract import TypelessOutputContract
    from agm.agl.runtime.request import AgentRequest, compose_agent_prompt

    dispatch = value_driven_agent_factory(idle_timeout=None)
    agent = agent_value("AgentCommand", command="runner")
    contract = TypelessOutputContract(
        target_type="int",
        codec_name="json",
        strict_json=True,
        format_instructions="Return JSON.",
        json_schema=None,
    )

    request = AgentRequest(
        agent=agent,
        prompt="Do X.",
        attempt=1,
        previous_invalid_output="bad",
        output_contract=contract,
    )
    request.prompt = compose_agent_prompt(request)
    dispatch(request)

    prompt = fake_agent_transport.calls[0][0]
    prompt_pos = prompt.index("Do X.")
    fmt_pos = prompt.index("Return JSON.")
    retry_pos = prompt.index("Your previous response")
    assert prompt_pos < fmt_pos < retry_pos


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


def test_agent_runner_gets_a_fresh_copy_of_the_host_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner mutations neither reach the host nor leak into a later agent call."""
    from agm.core.process import ProcessCaptureResult

    variable = "AGL_AGENT_HOST_ENV"
    monkeypatch.setenv(variable, "original")
    received: list[dict[str, str]] = []

    def fake_run_capture_result(
        command: list[str], *, env: dict[str, str] | None = None, **kwargs: object
    ) -> ProcessCaptureResult:
        del command, kwargs
        assert env is not None
        received.append(dict(env))
        env[variable] = "changed-by-runner"
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

    result = run_inline_command(
        PipelineDriver(agent_dispatcher=value_driven_agent_factory(idle_timeout=None)),
        'let first: text = ask("one", agent = AgentCommand("runner"))\n'
        'let second: text = ask("two", agent = AgentCommand("runner"))\n'
        "second",
    )

    assert result.ok, result.diagnostics
    assert [env[variable] for env in received] == ["original", "original"]
    assert os.environ[variable] == "original"


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

    run = run_inline_command(
        runtime,
        'let answer: text = ask("hello", '
        'agent = AgentCommand("runner --flag=\\%{AGM_NO_SUCH_VARIABLE}"))\nanswer',
    )

    assert not run.ok
    assert run.error is not None
    assert run.error.type_name == "AgentCallError"
    assert run.error.fields["cause"] == "spawn_failure"


def test_invalid_agent_value_becomes_typed_error() -> None:
    runtime = PipelineDriver(agent_dispatcher=value_driven_agent_factory(idle_timeout=None))

    run = run_inline_command(
        runtime,
        'let answer: text = ask("hello", agent = AgentCommand(""))\nanswer',
    )

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
    result = run_inline_command(
        runtime,
        "import std/config\n"
        'std/config::default-agent := AgentCommand("first")\n'
        'let first: text = ask("one")\n'
        'std/config::default-agent := AgentClaude("sonnet", "medium")\n'
        'let second: int = ask("two", on_parse_error = Retry(n = 0))\n'
        "second",
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
