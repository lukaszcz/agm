"""Acceptance tests for value-driven AgL Agent dispatch."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, get_type_hints

import pytest

from agm.agent.spec import (
    AGENT_SPECS,
    AgentClaude,
    AgentCodex,
    AgentCommand,
    AgentPi,
    payload_fields,
)
from agm.agl import PipelineDriver
from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
from agm.agl.runtime.agents import agent_value as encode_agent_value
from agm.agl.runtime.agents import decode_agent_value, value_driven_agent_factory
from agm.agl.semantics.type_table import BUILTIN_PRELUDE_TYPE_DEFS, create_seeded_type_table
from agm.agl.semantics.types import TextType
from agm.config.context import ConfigContext
from tests._agl_helpers import (
    agent_value,
    hermetic_get_sandbox_context,
    run_inline_command,
    session_sandbox_context,
    write_sandbox_home,
)
from tests.conftest import FakeAgentTransport

if TYPE_CHECKING:
    from agm.core.process import ProcessCaptureResult


def _agent_member_fields() -> dict[str, dict[str, object]]:
    table = create_seeded_type_table()
    return {
        member.name: dict(table.record_fields(member))
        for member in BUILTIN_PRELUDE_TYPE_DEFS["Agent"].members
    }


def test_host_specs_match_declared_agent_variants() -> None:
    """The host decoder catalog must track the checked ``Agent`` prelude shape."""
    declared = {variant: tuple(payload) for variant, payload in _agent_member_fields().items()}

    assert set(AGENT_SPECS) == set(declared)
    for variant, spec_cls in AGENT_SPECS.items():
        assert payload_fields(spec_cls) == declared[variant]
        hints = get_type_hints(spec_cls)
        assert all(hints[name] is str for name in payload_fields(spec_cls))
    assert all(
        isinstance(field_type, TextType)
        for payload in _agent_member_fields().values()
        for field_type in payload.values()
    )


def test_decode_accepts_every_declared_agent_variant() -> None:
    for variant, payload in _agent_member_fields().items():
        value = agent_value(variant, **{name: name for name in payload})

        assert isinstance(decode_agent_value(value, NO_BUILTIN_DECLARATIONS), AGENT_SPECS[variant])


def test_decode_rejects_declared_variant_without_a_host_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = dict(AGENT_SPECS)
    del catalog["AgentCommand"]
    monkeypatch.setattr("agm.agent.spec.AGENT_SPECS", catalog)

    with pytest.raises(ValueError):
        decode_agent_value(agent_value("AgentCommand", command="runner"), NO_BUILTIN_DECLARATIONS)


def test_agent_value_encodes_the_inverse_of_decode_agent_value() -> None:
    """``agent_value`` round-trips every declared ``Agent`` variant's host spec."""
    for variant, payload in _agent_member_fields().items():
        fields_by_name = {name: name for name in payload}
        spec = AGENT_SPECS[variant](**fields_by_name)

        value = encode_agent_value(spec, NO_BUILTIN_DECLARATIONS)

        assert decode_agent_value(value, NO_BUILTIN_DECLARATIONS) == spec
        assert value == agent_value(variant, **fields_by_name)


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
def test_ask_dispatches_each_agent_value_under_disabled_sandbox(
    fake_agent_transport: FakeAgentTransport, source: str, argv: list[str]
) -> None:
    """``AgentSandbox::Disabled`` reproduces the un-sandboxed argv byte for byte."""
    fake_agent_transport.queue(fake_agent_transport.success("ok"))
    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
        ),
        get_sandbox_context=None,
    )
    result = run_inline_command(
        runtime,
        f'let answer: text = ask("hello", agent = {source}, sandbox = AgentSandbox::Disabled)\n'
        "answer",
    )

    assert result.ok
    assert fake_agent_transport.calls == [("hello", argv)]


@pytest.mark.parametrize(
    ("source", "argv"),
    [
        ('AgentCommand("runner --flag")', ["runner", "--flag"]),
        (
            'AgentClaude("sonnet", "medium")',
            [
                "claude",
                "-p",
                "--model",
                "sonnet",
                "--effort",
                "medium",
                "--dangerously-skip-permissions",
            ],
        ),
        (
            'AgentCodex("o3", "high")',
            [
                "codex",
                "exec",
                "--model",
                "o3",
                "-c",
                "model_reasoning_effort=high",
                "--dangerously-bypass-approvals-and-sandbox",
                "-",
            ],
        ),
        (
            'AgentPi("openai", "gpt", "low")',
            ["pi", "-p", "--provider", "openai", "--model", "gpt", "--thinking", "low"],
        ),
    ],
)
def test_ask_dispatches_each_agent_value_selects_each_spec_permission_flag(
    fake_agent_transport: FakeAgentTransport, source: str, argv: list[str]
) -> None:
    """No ``sandbox`` operand defaults to ``Sandbox``: an all-permissions flag per spec.

    ``fake_agent_transport`` replaces ``prepare_rendered_prompt_run`` wholesale
    and records the argv *before* sandbox wrapping, so this only proves which
    permission flag each spec's argv carries under the default mode, never
    that the call is actually wrapped by the sandbox library --
    ``test_default_sandbox_mode_wraps_the_argv_and_honours_run_config_memory``
    (below) proves the wrapping itself, driving the real seam.
    """
    fake_agent_transport.queue(fake_agent_transport.success("ok"))
    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
        ),
        get_sandbox_context=None,
    )
    result = run_inline_command(
        runtime,
        f'let answer: text = ask("hello", agent = {source})\nanswer',
    )

    assert result.ok
    assert fake_agent_transport.calls == [("hello", argv)]


@pytest.mark.parametrize(
    ("sandbox_operand", "expected_argv"),
    [
        (
            "AgentSandbox::Disabled",
            ["claude", "-p", "--model", "sonnet", "--effort", "medium"],
        ),
        (
            "AgentSandbox::Native",
            [
                "claude",
                "-p",
                "--model",
                "sonnet",
                "--effort",
                "medium",
                "--permission-mode",
                "auto",
            ],
        ),
    ],
)
def test_disabled_and_native_sandbox_modes_never_wrap_the_argv(
    monkeypatch: pytest.MonkeyPatch, sandbox_operand: str, expected_argv: list[str]
) -> None:
    """``Disabled``/``Native`` never route through the sandbox library.

    Drives the real ``prepare_rendered_prompt_run`` seam (only
    ``run_capture_result`` is mocked, per ``tests/CLAUDE.md``), so the argv
    reaching the process primitive is observed directly: it starts with the
    agent binary itself, never ``systemd-run``/``srt``.
    """
    from agm.core.process import CapturedOutput, ProcessCaptureResult

    captured: dict[str, object] = {}

    def fake_run_capture_result(cmd: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured["cmd"] = cmd
        return ProcessCaptureResult(
            returncode=0,
            stdout=CapturedOutput(data=b"ok", truncated=False),
            stderr=CapturedOutput(data=b"", truncated=False),
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
        )

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)

    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
        ),
        get_sandbox_context=None,
    )
    result = run_inline_command(
        runtime,
        'let answer: text = ask("hello", agent = AgentClaude("sonnet", "medium"), '
        f"sandbox = {sandbox_operand})\nanswer",
    )

    assert result.ok, result.diagnostics
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "claude"
    assert "systemd-run" not in cmd
    assert "srt" not in cmd
    # The prompt file target is appended after the spec's own argv.
    assert cmd[:-1] == expected_argv
    assert cmd[-1].startswith("@")


@pytest.mark.parametrize(
    ("source", "argv"),
    [
        (
            'AgentClaude("sonnet", "medium")',
            [
                "claude",
                "-p",
                "--model",
                "sonnet",
                "--effort",
                "medium",
                "--permission-mode",
                "auto",
            ],
        ),
        (
            'AgentCodex("o3", "high")',
            [
                "codex",
                "exec",
                "--model",
                "o3",
                "-c",
                "model_reasoning_effort=high",
                "--approve-for-me",
                "-",
            ],
        ),
    ],
)
def test_ask_dispatches_claude_and_codex_under_native_sandbox_mode(
    fake_agent_transport: FakeAgentTransport, source: str, argv: list[str]
) -> None:
    """``AgentSandbox::Native`` selects each agent's own all-permissions flag, unsandboxed."""
    fake_agent_transport.queue(fake_agent_transport.success("ok"))
    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
        ),
        get_sandbox_context=None,
    )
    result = run_inline_command(
        runtime,
        f'let answer: text = ask("hello", agent = {source}, sandbox = AgentSandbox::Native)\n'
        "answer",
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
    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
        ),
        get_sandbox_context=None,
    )

    run = run_inline_command(
        runtime,
        'let answer: text = ask("hello", agent = AgentCommand("runner"))\nanswer',
    )

    assert not run.ok
    assert run.error is not None
    assert run.error.type_name == "AgentCallError"
    assert run.error.fields["agent"] == {"$case": "AgentCommand", "command": "runner"}


def test_undecodable_agent_stdout_becomes_a_protocol_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An otherwise-successful run whose stdout is not valid UTF-8 is a protocol failure."""
    from agm.core.process import CapturedOutput, ProcessCaptureResult

    def fake_run_capture_result(cmd: list[str], **kwargs: object) -> ProcessCaptureResult:
        return ProcessCaptureResult(
            returncode=0,
            stdout=CapturedOutput(data=b"ok \xff bad", truncated=False),
            stderr=CapturedOutput(data=b"", truncated=False),
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
        )

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)
    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
        ),
        get_sandbox_context=None,
    )

    # Disabled: this test drives the real ``prepare_rendered_prompt_run`` seam
    # (only ``run_capture_result`` is mocked), so a real sandbox mode would try
    # to resolve ``srt`` on PATH -- irrelevant to the protocol-failure assertion.
    run = run_inline_command(
        runtime,
        "let answer: text = "
        'ask("hello", agent = AgentCommand("runner"), sandbox = AgentSandbox::Disabled)\n'
        "answer",
    )

    assert not run.ok
    assert run.error is not None
    assert run.error.type_name == "AgentCallError"
    assert run.error.fields["cause"] == "protocol_failure"
    # The message locates the first invalid byte, which the cause alone cannot.
    assert "byte 3" in str(run.error.fields["message"])


def test_caught_agent_call_error_keeps_static_agent_encoding_when_raised_later(
    fake_agent_transport: FakeAgentTransport,
) -> None:
    """An AgentCallError remains statically encoded after catch, storage, and re-raise."""
    fake_agent_transport.queue(
        fake_agent_transport.failure(returncode=2, stderr="boom", elapsed=1.0)
    )
    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
        ),
        get_sandbox_context=None,
    )

    run = run_inline_command(
        runtime,
        "let saved = try\n"
        '  let _ = ask("hello", agent = AgentCommand("runner"))\n'
        '  raise Abort(message = "unreachable")\n'
        "catch AgentCallError as err => err\n"
        "raise saved",
    )

    assert not run.ok
    assert run.error is not None
    assert run.error.type_name == "AgentCallError"
    assert run.error.fields["agent"] == {"$case": "AgentCommand", "command": "runner"}


def test_user_exception_enum_field_keeps_slot_encoding_after_storage_and_reraise(
    fake_agent_transport: FakeAgentTransport,
) -> None:
    """User exception provenance is nominal-keyed, not reserved for host errors."""
    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
        ),
        get_sandbox_context=None,
    )

    run = run_inline_command(
        runtime,
        "enum Status | Open | Closed\n"
        "exception Problem extends Exception\n"
        "  status: Status\n"
        "let saved = try\n"
        '  raise Problem(message = "bad", status = Closed)\n'
        "catch Problem as err => err\n"
        "raise saved",
    )

    assert not run.ok
    assert run.error is not None
    assert run.error.type_name == "Problem"
    assert run.error.fields["status"] == {"$case": "Closed"}


def test_nonzero_exit_message_includes_the_exit_code(
    fake_agent_transport: FakeAgentTransport,
) -> None:
    """A caught ``AgentCallError.message`` still names the exit code (not just metadata)."""
    fake_agent_transport.queue(
        fake_agent_transport.failure(returncode=7, stderr="boom", elapsed=1.0)
    )
    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
        ),
        get_sandbox_context=None,
    )

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

    dispatch = value_driven_agent_factory(
        idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
    )
    agent = AgentCommand(command="runner")

    dispatch(AgentRequest(agent=agent, prompt="Do X."))

    assert fake_agent_transport.calls == [("Do X.", ["runner"])]


def test_composed_prompt_appends_format_instructions_after_the_prompt(
    fake_agent_transport: FakeAgentTransport,
) -> None:
    from agm.agl.runtime.contract import TypelessOutputContract
    from agm.agl.runtime.request import AgentRequest, compose_agent_prompt

    dispatch = value_driven_agent_factory(
        idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
    )
    agent = AgentCommand(command="runner")
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

    dispatch = value_driven_agent_factory(
        idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
    )
    agent = AgentCommand(command="runner")
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

    dispatch = value_driven_agent_factory(
        idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
    )
    agent = AgentCommand(command="runner")

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
    assert "- The response is missing required data." in prompt
    assert "- The response contains a value with an incorrect type." in prompt
    assert "the-bad-output-xyz" in prompt
    assert "Return only valid JSON matching the schema." in prompt


def test_composed_prompt_has_no_retry_feedback_on_the_first_attempt(
    fake_agent_transport: FakeAgentTransport,
) -> None:
    from agm.agl.runtime.request import AgentRequest

    dispatch = value_driven_agent_factory(
        idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
    )
    agent = AgentCommand(command="runner")

    dispatch(AgentRequest(agent=agent, prompt="Do X.", attempt=0))

    prompt = fake_agent_transport.calls[0][0]
    assert "Your previous response did not match" not in prompt


def test_composed_prompt_orders_format_instructions_before_retry_feedback(
    fake_agent_transport: FakeAgentTransport,
) -> None:
    """Ordering: prompt, then format_instructions, then the retry-feedback block."""
    from agm.agl.runtime.contract import TypelessOutputContract
    from agm.agl.runtime.request import AgentRequest, compose_agent_prompt

    dispatch = value_driven_agent_factory(
        idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
    )
    agent = AgentCommand(command="runner")
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
    from agm.core.process import CapturedOutput, ProcessCaptureResult

    captured: dict[str, object] = {}

    def fake_run_capture_result(cmd: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured["cmd"] = cmd
        captured["stdin_text"] = kwargs.get("stdin_text")
        return ProcessCaptureResult(
            returncode=0,
            stdout=CapturedOutput(data=b"ok", truncated=False),
            stderr=CapturedOutput(data=b"", truncated=False),
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
        )

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)

    agent = AgentCodex(model="o3", thinking="high")
    dispatch = value_driven_agent_factory(
        idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
    )

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
    from agm.core.process import CapturedOutput, ProcessCaptureResult

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
            stdout=CapturedOutput(data=b"ok", truncated=False),
            stderr=CapturedOutput(data=b"", truncated=False),
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
        )

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)

    result = run_inline_command(
        PipelineDriver(
            agent_dispatcher=value_driven_agent_factory(
                idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
            ),
            get_sandbox_context=None,
        ),
        'let first: text = ask("one", agent = AgentCommand("runner"), '
        "sandbox = AgentSandbox::Disabled)\n"
        'let second: text = ask("two", agent = AgentCommand("runner"), '
        "sandbox = AgentSandbox::Disabled)\n"
        "second",
    )

    assert result.ok, result.diagnostics
    assert [env[variable] for env in received] == ["original", "original"]
    assert os.environ[variable] == "original"


def test_unresolvable_command_hole_becomes_a_typed_error() -> None:
    """A host interpolation hole the environment cannot fill fails as a spawn failure."""
    from agm.agl.runtime.agents import AgentCallHostError
    from agm.agl.runtime.request import AgentRequest

    agent = AgentCommand(command="runner --flag=%{AGM_NO_SUCH_VARIABLE}")
    dispatch = value_driven_agent_factory(
        idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
    )

    with pytest.raises(AgentCallHostError) as exc_info:
        dispatch(AgentRequest(agent=agent, prompt="hello"))

    assert exc_info.value.cause == "spawn_failure"


def test_escaped_command_hole_reaches_the_host_interpolator() -> None:
    """`\\%{` in AgL source passes the hole through for the host to resolve."""
    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
        ),
        get_sandbox_context=None,
    )

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
    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=hermetic_get_sandbox_context()
        ),
        get_sandbox_context=None,
    )

    run = run_inline_command(
        runtime,
        'let answer: text = ask("hello", agent = AgentCommand(""))\nanswer',
    )

    assert not run.ok
    assert run.error is not None
    assert run.error.type_name == "AgentCallError"


def test_default_agent_value_is_read_at_each_call_and_errors_stay_typed() -> None:
    from agm.agl.runtime.request import AgentRequest

    requests: list[AgentCommand | AgentClaude | AgentCodex | AgentPi] = []

    def agent(request: AgentRequest) -> str:
        value = request.agent
        assert isinstance(value, (AgentCommand, AgentClaude, AgentCodex, AgentPi))
        requests.append(value)
        return "not an integer"

    runtime = PipelineDriver(agent_dispatcher=agent, get_sandbox_context=None)
    result = run_inline_command(
        runtime,
        "import std/config\n"
        'std/config::default-agent := AgentCommand("first")\n'
        'let first: text = ask("one")\n'
        'std/config::default-agent := AgentClaude("sonnet", "medium")\n'
        'let second: int = ask("two", on-parse-error = Retry(n = 0))\n'
        "second",
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "AgentParseError"
    assert result.error.fields["agent"] == {
        "$case": "AgentCommand",
        "command": "first",
    }
    assert [type(request).__name__ for request in requests] == [
        "AgentCommand",
        "AgentCommand",
    ]


def _capturing_run_capture_result(
    captured: dict[str, object], *, returncode: int = 0
) -> Callable[..., "ProcessCaptureResult"]:
    """A fake ``run_capture_result`` that records the argv it received and returns *returncode*."""
    from agm.core.process import CapturedOutput, ProcessCaptureResult

    def fake(cmd: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return ProcessCaptureResult(
            returncode=returncode,
            stdout=CapturedOutput(data=b"ok", truncated=False),
            stderr=CapturedOutput(data=b"", truncated=False),
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
        )

    return fake


@pytest.mark.parametrize(
    ("source", "profile", "argv_prefix"),
    [
        ('AgentClaude("sonnet", "medium")', "claude", ["claude", "-p"]),
        ('AgentCodex("o3", "high")', "codex", ["codex", "exec"]),
        ('AgentPi("openai", "gpt", "low")', "pi", ["pi", "-p"]),
        ('AgentCommand("/some/path/my-agent")', "my-agent", ["/some/path/my-agent"]),
    ],
)
def test_default_sandbox_mode_wraps_the_argv_and_honours_run_config_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    profile: str,
    argv_prefix: list[str],
) -> None:
    """The default ``Sandbox`` mode reaches the agent binary through the sandbox
    library for every agent spec kind: ``[run.<profile>].memory`` is honoured
    under the real profile name -- the spec's own executable, or an
    ``AgentCommand``'s basename -- and ``.alias`` is never consulted for agent
    argv (aliases apply only to ``agm run``). Also gives codex-under-sandbox
    its first coverage: it delivers its prompt on stdin, which must still
    pass through the wrapper prefix.
    """
    home = tmp_path / "home"
    alias = f"not-{profile}"
    write_sandbox_home(
        home,
        run_toml=f'[run.{profile}]\nmemory = "4G"\nalias = "{alias}"\n',
        # A settings file matching the alias: if the alias were (wrongly)
        # honoured, it would be selected over the unqualified default below.
        extra_settings_files=(alias,),
    )
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "agm.agent.runner.run_capture_result", _capturing_run_capture_result(captured)
    )

    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=session_sandbox_context(home)
        ),
        get_sandbox_context=None,
    )
    result = run_inline_command(
        runtime,
        f'let answer: text = ask("hello", agent = {source})\nanswer',
    )

    assert result.ok, result.diagnostics
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:4] == ["systemd-run", "--user", "--scope", "-q"]
    assert "MemoryMax=4G" in cmd
    srt_index = cmd.index("srt")
    assert cmd[srt_index + 2] == str(home / ".agm" / "sandbox" / "default.json")
    tail = cmd[srt_index + 4 :]
    assert tail[: len(argv_prefix)] == argv_prefix


def test_interpolated_agent_command_sandboxes_under_the_real_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sandbox profile follows the post-interpolation argv, not the template.

    ``AgentCommand("%{TOOL}/bin/agent")`` must sandbox under ``agent`` (the
    real executable that runs), never under the literal ``%{TOOL}`` template
    fragment: ``command_with_prompt_target`` interpolates the argv *after*
    the spec's own ``argv()`` builds it, so the profile can only be known
    correctly once that interpolation has happened.
    """
    home = tmp_path / "home"
    write_sandbox_home(home, run_toml='[run.agent]\nmemory = "2G"\n')
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")
    monkeypatch.setenv("TOOL", str(tmp_path / "tools"))

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "agm.agent.runner.run_capture_result", _capturing_run_capture_result(captured)
    )

    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=session_sandbox_context(home)
        ),
        get_sandbox_context=None,
    )
    result = run_inline_command(
        runtime,
        'let answer: text = ask("hello", agent = AgentCommand("\\%{TOOL}/bin/agent"))\nanswer',
    )

    assert result.ok, result.diagnostics
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    # The interpolated real executable's memory config was honoured...
    assert "MemoryMax=2G" in cmd
    srt_index = cmd.index("srt")
    tail = cmd[srt_index + 4 :]
    # ...and the settings-selection probe used the real basename, not the
    # unresolved "%{TOOL}" template fragment.
    assert tail[0] == str(tmp_path / "tools" / "bin" / "agent")


def test_agent_call_info_argv_is_the_wrapped_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``AgentCallInfo.argv`` records what actually ran -- the sandboxed argv,
    not the spec's own pre-wrap argv."""
    from agm.agent.transport import AgentCallInfo
    from agm.agl.runtime.request import AgentRequest
    from agm.sandbox.request import SandboxLimits

    home = tmp_path / "home"
    write_sandbox_home(home)
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "agm.agent.runner.run_capture_result", _capturing_run_capture_result(captured)
    )

    dispatch = value_driven_agent_factory(
        idle_timeout=None, get_sandbox_context=session_sandbox_context(home)
    )

    response = dispatch(
        AgentRequest(
            agent=AgentClaude("sonnet", "medium"),
            prompt="hello",
            sandbox=SandboxLimits(),
        )
    )

    assert not isinstance(response, str)
    call_info = response.call_info
    assert isinstance(call_info, AgentCallInfo)
    assert call_info.argv == captured["cmd"]
    assert call_info.argv[0] == "systemd-run"
    assert call_info.sandboxed is True


def test_sandbox_context_is_built_at_most_once_per_get_sandbox_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `[run.*]` config load behind a `get_sandbox_context` callable runs
    at most once no matter how many sandboxed calls dispatch through it --
    including across every factory built from the same callable, the sharing
    `agm exec`/`agm repl` rely on -- and never runs at all when nothing
    dispatches through it."""
    from agm.config import general as config_general
    from agm.sandbox.prepare import lazy_sandbox_context

    home = tmp_path / "home"
    write_sandbox_home(home)
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")
    monkeypatch.setattr("agm.agent.runner.run_capture_result", _capturing_run_capture_result({}))

    load_calls: list[int] = []
    real_load_run_config = config_general.load_run_config

    def counting_load_run_config(**kwargs: object) -> object:
        load_calls.append(1)
        return real_load_run_config(**kwargs)

    monkeypatch.setattr(config_general, "load_run_config", counting_load_run_config)

    get_sandbox_context = lazy_sandbox_context(ConfigContext(home=home, proj_dir=None, cwd=home))

    # Built but never dispatched: the config load never runs.
    value_driven_agent_factory(idle_timeout=None, get_sandbox_context=get_sandbox_context)
    assert load_calls == []

    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=get_sandbox_context
        ),
        get_sandbox_context=None,
    )
    result = run_inline_command(
        runtime,
        'let a: text = ask("one", agent = AgentClaude("sonnet", "medium"))\n'
        'let b: text = ask("two", agent = AgentClaude("sonnet", "medium"))\n'
        "b",
    )

    assert result.ok, result.diagnostics
    assert load_calls == [1]


def test_explicit_settings_file_selects_that_file_over_the_profile_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Sandbox(settings = Some(path))`` replaces the default settings-file chain."""
    home = tmp_path / "home"
    write_sandbox_home(home)
    explicit_settings = tmp_path / "explicit.json"
    explicit_settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "agm.agent.runner.run_capture_result", _capturing_run_capture_result(captured)
    )

    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=session_sandbox_context(home)
        ),
        get_sandbox_context=None,
    )
    result = run_inline_command(
        runtime,
        'let answer: text = ask("hello", agent = AgentClaude("sonnet", "medium"), '
        f'sandbox = Sandbox(settings = Some("{explicit_settings.as_posix()}")))\nanswer',
    )

    assert result.ok, result.diagnostics
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    srt_index = cmd.index("srt")
    assert cmd[srt_index : srt_index + 3] == ["srt", "--settings", str(explicit_settings)]


def test_missing_srt_becomes_an_agent_call_error_carrying_the_library_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``srt`` absent from PATH surfaces as ``AgentCallError`` with the library text."""
    home = tmp_path / "home"
    write_sandbox_home(home)

    def fake_which(name: str, *args: object, **kwargs: object) -> str | None:
        return "/usr/bin/systemd-run" if name == "systemd-run" else None

    monkeypatch.setattr("shutil.which", fake_which)

    def fail_run_capture_result(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not spawn a process when sandbox preparation failed")

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fail_run_capture_result)

    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=session_sandbox_context(home)
        ),
        get_sandbox_context=None,
    )
    result = run_inline_command(
        runtime,
        'let answer: text = ask("hello", agent = AgentClaude("sonnet", "medium"))\nanswer',
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.type_name == "AgentCallError"
    assert result.error.fields["cause"] == "spawn_failure"
    metadata = result.error.fields["metadata"]
    assert isinstance(metadata, dict)
    assert "srt is not installed" in str(metadata["stderr_tail"])


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_temp_settings_cleanup_runs_after_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """A per-call temp settings file (written when patching for a project dir) is
    removed whether the dispatched call succeeds or fails.

    Mocks only the external boundary (``run_capture_result``, per
    ``tests/CLAUDE.md``): the settings path under test is read back out of
    the captured argv, rather than monkeypatching the sandbox library's own
    private ``_write_json_temp`` helper.
    """
    home = tmp_path / "home"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    write_sandbox_home(home)
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "agm.agent.runner.run_capture_result",
        _capturing_run_capture_result(captured, returncode=0 if outcome == "success" else 1),
    )

    runtime = PipelineDriver(
        agent_dispatcher=value_driven_agent_factory(
            idle_timeout=None, get_sandbox_context=session_sandbox_context(home, proj_dir=proj_dir)
        ),
        get_sandbox_context=None,
    )
    result = run_inline_command(
        runtime,
        'let answer: text = ask("hello", agent = AgentClaude("sonnet", "medium"))\nanswer',
    )

    assert result.ok is (outcome == "success")

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    srt_index = cmd.index("srt")
    settings_path = Path(cmd[srt_index + 2])
    # The project-dir patch step always writes a fresh temp settings file --
    # never the home's own unqualified `default.json` -- so this is that
    # per-call temp file.
    assert settings_path != home / ".agm" / "sandbox" / "default.json"
    assert not settings_path.exists()
