"""Unit tests for native CLI-continuation agent session adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from agm.agent.session import (
    SessionAskError,
    SessionAskRequest,
    SessionBackend,
    SessionHostError,
    SessionOpenRequest,
    SessionOperation,
    SessionService,
)
from agm.agent.session.cli_adapters import (
    CLI_SESSION_BACKENDS,
    AgentCommandSessionBackend,
    ClaudeCliSessionBackend,
    CodexCliSessionBackend,
    PiCliSessionBackend,
)
from agm.agent.spec import AGENT_SPECS, AgentClaude, AgentCodex, AgentCommand, AgentPi, AgentSpec
from agm.agl.semantics.type_table import BUILTIN_PRELUDE_TYPE_DEFS
from agm.core.process import ProcessCaptureResult


@dataclass(frozen=True)
class CaptureOutcome:
    """One subprocess result returned by :class:`CaptureTransport`."""

    stdout: str = "answer"
    returncode: int | None = 0
    stderr: str = ""
    timed_out: bool = False
    spawn_error: str | None = None


class CaptureTransport:
    """The runner subprocess seam, recording argv and stdin without an agent CLI."""

    def __init__(self, outcomes: list[CaptureOutcome]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[list[str], str | None]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def run(argv: list[str], **kwargs: object) -> ProcessCaptureResult:
            stdin_text = kwargs.get("stdin_text")
            self.calls.append((argv, stdin_text if isinstance(stdin_text, str) else None))
            outcome = self.outcomes.pop(0)
            return ProcessCaptureResult(
                returncode=outcome.returncode,
                stdout=outcome.stdout,
                stderr=outcome.stderr,
                elapsed=0.1,
                timed_out=outcome.timed_out,
                spawn_error=outcome.spawn_error,
                spawn_errno=None,
            )

        monkeypatch.setattr("agm.agent.runner.run_capture_result", run)


def _open(backend: object, agent: object, *, name: str = "", one_shot: bool = False) -> None:
    if not isinstance(
        backend,
        (
            AgentCommandSessionBackend,
            ClaudeCliSessionBackend,
            CodexCliSessionBackend,
            PiCliSessionBackend,
        ),
    ):
        raise AssertionError("unexpected backend")
    backend.open(SessionOpenRequest(agent=agent, transport="cli", name=name, one_shot=one_shot))


def _file_prompt_argv(argv: list[str], command: list[str]) -> None:
    """Assert normal file delivery without pinning its temporary path."""
    assert argv[:-1] == command
    assert argv[-1].startswith("@")


def test_prepared_runner_rejects_conflicting_delivery_options() -> None:
    from agm.agent.runner import PromptDelivery, prepare_rendered_prompt_run

    with pytest.raises(ValueError):
        prepare_rendered_prompt_run(
            "prompt",
            runner=["runner"],
            temp_files=[],
            env={},
            prompt_via_stdin=True,
            delivery=PromptDelivery.LITERAL,
        )


def test_agent_variant_spec_backend_catalogs_are_in_lockstep() -> None:
    expected: dict[str, tuple[tuple[str, ...], type[AgentSpec], type[SessionBackend]]] = {
        "AgentCommand": (("command",), AgentCommand, AgentCommandSessionBackend),
        "AgentClaude": (("model", "thinking"), AgentClaude, ClaudeCliSessionBackend),
        "AgentCodex": (("model", "thinking"), AgentCodex, CodexCliSessionBackend),
        "AgentPi": (("provider", "model", "thinking"), AgentPi, PiCliSessionBackend),
    }
    declared = dict(BUILTIN_PRELUDE_TYPE_DEFS["Agent"].variants)

    assert set(declared) == set(expected) == set(AGENT_SPECS) == set(CLI_SESSION_BACKENDS)
    for variant, (fields, spec, backend) in expected.items():
        assert tuple(name for name, _ in declared[variant]) == fields
        assert spec.PAYLOAD_FIELDS == fields
        assert AGENT_SPECS[variant] is spec
        assert CLI_SESSION_BACKENDS[variant] is backend


def test_claude_delivers_compact_literal_and_fork_promptlessly_through_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome("first"),
            CaptureOutcome('{"is_error": false}'),
            CaptureOutcome('{"is_error": false}'),
            CaptureOutcome('{"session_id": "child"}'),
        ]
    )
    transport.install(monkeypatch)
    agent = AgentClaude("m", "t")
    backend = ClaudeCliSessionBackend()
    _open(backend, agent, name="named")

    backend.ask(SessionAskRequest("first"))
    backend.compact("instructions")
    backend.compact("")
    child = backend.fork()

    first_id = transport.calls[0][0][3]
    _file_prompt_argv(
        transport.calls[0][0],
        [
            "claude",
            "-p",
            "--session-id",
            first_id,
            "-n",
            "named",
            "--model",
            "m",
            "--effort",
            "t",
        ],
    )
    assert transport.calls[0][1] is None
    assert transport.calls[1] == (
        [
            "claude",
            "-p",
            "--resume",
            first_id,
            "--output-format",
            "json",
            "--model",
            "m",
            "--effort",
            "t",
            "/compact instructions",
        ],
        None,
    )
    assert transport.calls[2] == (
        [
            "claude",
            "-p",
            "--resume",
            first_id,
            "--output-format",
            "json",
            "--model",
            "m",
            "--effort",
            "t",
            "/compact",
        ],
        None,
    )
    assert transport.calls[3] == (
        [
            "claude",
            "-p",
            "--resume",
            first_id,
            "--fork-session",
            "--output-format",
            "json",
            "--model",
            "m",
            "--effort",
            "t",
        ],
        None,
    )
    assert isinstance(child, ClaudeCliSessionBackend)


def test_claude_forks_immediately_after_open_with_independent_child_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome('{"session_id": "child"}'),
            CaptureOutcome("child answer"),
            CaptureOutcome("parent answer"),
        ]
    )
    transport.install(monkeypatch)
    parent = ClaudeCliSessionBackend()
    _open(parent, AgentClaude("m", "t"), name="named")

    child = parent.fork()
    assert child.ask(SessionAskRequest("child follow-up")).content == "child answer"
    assert parent.ask(SessionAskRequest("parent follow-up")).content == "parent answer"

    parent_id = transport.calls[0][0][3]
    assert parent_id != "child"
    assert transport.calls[0] == (
        [
            "claude",
            "-p",
            "--resume",
            parent_id,
            "--fork-session",
            "--output-format",
            "json",
            "--model",
            "m",
            "--effort",
            "t",
        ],
        None,
    )
    _file_prompt_argv(
        transport.calls[1][0],
        [
            "claude",
            "-p",
            "--resume",
            "child",
            "--model",
            "m",
            "--effort",
            "t",
        ],
    )
    _file_prompt_argv(
        transport.calls[2][0],
        [
            "claude",
            "-p",
            "--session-id",
            parent_id,
            "-n",
            "named",
            "--model",
            "m",
            "--effort",
            "t",
        ],
    )


def test_pi_forks_immediately_after_open_then_child_is_live_without_repeating_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [CaptureOutcome("forked"), CaptureOutcome("child"), CaptureOutcome("parent")]
    )
    transport.install(monkeypatch)
    agent = AgentPi("p", "m", "t")
    parent = PiCliSessionBackend()
    _open(parent, agent, name="named")

    child = parent.fork()
    child.ask(SessionAskRequest("child"))
    parent.ask(SessionAskRequest("parent"))

    child_id = transport.calls[0][0][3]
    parent_id = transport.calls[0][0][5]
    assert child_id != parent_id
    assert transport.calls[0] == (
        [
            "pi",
            "-p",
            "--session-id",
            child_id,
            "--fork",
            parent_id,
            "--provider",
            "p",
            "--model",
            "m",
            "--thinking",
            "t",
        ],
        None,
    )
    _file_prompt_argv(
        transport.calls[1][0],
        [
            "pi",
            "-p",
            "--session-id",
            child_id,
            "--provider",
            "p",
            "--model",
            "m",
            "--thinking",
            "t",
        ],
    )
    _file_prompt_argv(
        transport.calls[2][0],
        [
            "pi",
            "-p",
            "--session-id",
            parent_id,
            "--name",
            "named",
            "--provider",
            "p",
            "--model",
            "m",
            "--thinking",
            "t",
        ],
    )


@pytest.mark.parametrize(
    ("backend", "agent", "name_flag"),
    [
        (ClaudeCliSessionBackend, AgentClaude("m", "t"), "-n"),
        (PiCliSessionBackend, AgentPi("p", "m", "t"), "--name"),
    ],
)
def test_session_id_cli_backends_reuse_ids_and_names_after_reset(
    monkeypatch: pytest.MonkeyPatch,
    backend: Callable[[], ClaudeCliSessionBackend | PiCliSessionBackend],
    agent: AgentClaude | AgentPi,
    name_flag: str,
) -> None:
    transport = CaptureTransport(
        [CaptureOutcome("first"), CaptureOutcome("second"), CaptureOutcome("reset")]
    )
    transport.install(monkeypatch)
    session = backend()
    _open(session, agent, name="named")

    session.ask(SessionAskRequest("first"))
    session.ask(SessionAskRequest("second"))
    session.reset()
    session.ask(SessionAskRequest("after reset"))

    first_argv, second_argv, reset_argv = (argv for argv, _ in transport.calls)
    first_id = first_argv[first_argv.index("--session-id") + 1]
    reset_id = reset_argv[reset_argv.index("--session-id") + 1]
    assert first_id in second_argv
    assert reset_id != first_id
    assert first_argv[first_argv.index(name_flag) + 1] == "named"
    assert name_flag not in second_argv
    assert reset_argv[reset_argv.index(name_flag) + 1] == "named"


def test_codex_one_shot_session_uses_the_standard_command(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = CaptureTransport([CaptureOutcome("answer")])
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("m", "t"), one_shot=True)

    assert backend.ask(SessionAskRequest("prompt")).content == "answer"
    assert transport.calls == [
        (
            ["codex", "exec", "--model", "m", "-c", "model_reasoning_effort=t", "-"],
            "prompt",
        )
    ]


def test_codex_reset_after_successful_creation_starts_a_new_jsonl_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome(
                '{"type":"thread.started","thread_id":"first"}\n'
                '{"type":"item.completed","item":'
                '{"type":"agent_message","text":"first answer"}}'
            ),
            CaptureOutcome(
                '{"type":"thread.started","thread_id":"second"}\n'
                '{"type":"item.completed","item":'
                '{"type":"agent_message","text":"second answer"}}'
            ),
        ]
    )
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("m", "t"))

    assert backend.ask(SessionAskRequest("first prompt")).content == "first answer"
    backend.reset()
    assert backend.ask(SessionAskRequest("second prompt")).content == "second answer"

    assert transport.calls == [
        (
            [
                "codex",
                "exec",
                "--json",
                "--model",
                "m",
                "-c",
                "model_reasoning_effort=t",
                "-",
            ],
            "first prompt",
        ),
        (
            [
                "codex",
                "exec",
                "--json",
                "--model",
                "m",
                "-c",
                "model_reasoning_effort=t",
                "-",
            ],
            "second prompt",
        ),
    ]


def test_codex_initial_ask_parses_jsonl_and_resume_returns_plaintext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome(
                "\n".join(
                    (
                        '{"type":"thread.started","thread_id":"first"}',
                        '{"type":"turn.started"}',
                        '{"type":"item.completed","item":'
                        '{"type":"agent_message","text":"first answer"}}',
                        '{"type":"turn.completed"}',
                    )
                )
            ),
            CaptureOutcome("later answer"),
        ]
    )
    transport.install(monkeypatch)
    agent = AgentCodex("m", "t")
    backend = CodexCliSessionBackend()
    _open(backend, agent)

    assert backend.ask(SessionAskRequest("first")).content == "first answer"
    assert backend.ask(SessionAskRequest("later")).content == "later answer"

    assert transport.calls == [
        (
            [
                "codex",
                "exec",
                "--json",
                "--model",
                "m",
                "-c",
                "model_reasoning_effort=t",
                "-",
            ],
            "first",
        ),
        (
            [
                "codex",
                "exec",
                "resume",
                "first",
                "--model",
                "m",
                "-c",
                "model_reasoning_effort=t",
                "-",
            ],
            "later",
        ),
    ]


@pytest.mark.parametrize(
    "output",
    [
        "not json",
        '{"type":"thread.started","thread_id":42}',
        '{"type":"other","thread_id":"wrong"}',
        '{"type":"thread.started","thread_id":"first"}\n[]',
        '{"type":"thread.started","thread_id":"first"}',
        '{"type":"thread.started","thread_id":"first"}\n{"type":"item.completed","item":{"type":"agent_message"}}',
        '{"thread_id":"first"}',
        '{"type":"thread.started","thread_id":"first"}\n{"type":"item.completed","item":[]}',
        '{"type":"thread.started","thread_id":"first"}\n{"type":"item.completed","item":{}}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}',
    ],
)
def test_codex_rejects_malformed_or_incomplete_jsonl(
    monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    transport = CaptureTransport([CaptureOutcome(output)])
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("", ""))

    with pytest.raises(SessionHostError) as raised:
        backend.ask(SessionAskRequest("first"))

    assert raised.value.operation == SessionOperation.ASK


def test_codex_ignores_completed_non_assistant_items(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome(
                "\n".join(
                    (
                        '{"type":"thread.started","thread_id":"first"}',
                        '{"type":"item.completed","item":{"type":"tool_call"}}',
                        '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}',
                    )
                )
            )
        ]
    )
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("", ""))

    assert backend.ask(SessionAskRequest("first")).content == "answer"


def test_codex_defers_thread_creation_until_the_first_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome(
                "\n".join(
                    (
                        '{"type":"thread.started","thread_id":"first"}',
                        '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}',
                    )
                )
            )
        ]
    )
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("model", "high"))

    for operation in (lambda: backend.compact(""), backend.fork, backend.stats):
        with pytest.raises(SessionHostError):
            operation()
    with pytest.raises(SessionHostError):
        backend.set_name("unsupported")
    backend.reset()
    assert transport.calls == []

    assert backend.ask(SessionAskRequest("create")).content == "answer"
    assert len(transport.calls) == 1
    for operation in (lambda: backend.compact(""), backend.fork, backend.stats):
        with pytest.raises(SessionHostError):
            operation()
    with pytest.raises(SessionHostError):
        backend.set_name("unsupported")
    backend.close()


def test_first_invocation_transport_failure_consumes_creation_state_until_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome(returncode=1, stderr="failed"),
            CaptureOutcome("resumed"),
            CaptureOutcome("reset"),
        ]
    )
    transport.install(monkeypatch)
    agent = AgentClaude("m", "t")
    backend = ClaudeCliSessionBackend()
    _open(backend, agent, name="named")

    with pytest.raises(SessionAskError):
        backend.ask(SessionAskRequest("first"))
    backend.ask(SessionAskRequest("retry"))
    backend.reset()
    backend.ask(SessionAskRequest("after reset"))

    original_id = transport.calls[0][0][3]
    _file_prompt_argv(
        transport.calls[0][0],
        [
            "claude",
            "-p",
            "--session-id",
            original_id,
            "-n",
            "named",
            "--model",
            "m",
            "--effort",
            "t",
        ],
    )
    _file_prompt_argv(
        transport.calls[1][0],
        [
            "claude",
            "-p",
            "--resume",
            original_id,
            "--model",
            "m",
            "--effort",
            "t",
        ],
    )
    reset_id = transport.calls[2][0][3]
    assert reset_id != original_id
    _file_prompt_argv(
        transport.calls[2][0],
        [
            "claude",
            "-p",
            "--session-id",
            reset_id,
            "-n",
            "named",
            "--model",
            "m",
            "--effort",
            "t",
        ],
    )


@pytest.mark.parametrize("operation", ["compact", "fork"])
def test_service_maps_cli_lifecycle_transport_failures_to_host_errors(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    transport = CaptureTransport([CaptureOutcome(returncode=1, stderr="failed")])
    transport.install(monkeypatch)
    backend = ClaudeCliSessionBackend()
    service = SessionService(lambda _agent, _transport: backend)
    handle = service.open(AgentClaude("", ""), "cli")

    with pytest.raises(SessionHostError) as raised:
        if operation == "compact":
            service.compact(handle)
        else:
            service.fork(handle)

    assert raised.value.operation == operation


@pytest.mark.parametrize(
    ("method", "output"),
    [
        ("compact", "not json"),
        ("compact", "[]"),
        ("compact", "{}"),
        ("fork", "not json"),
        ("fork", "[]"),
        ("fork", "{}"),
    ],
)
def test_claude_lifecycle_protocol_errors_are_host_errors(
    monkeypatch: pytest.MonkeyPatch, method: str, output: str
) -> None:
    transport = CaptureTransport([CaptureOutcome(output)])
    transport.install(monkeypatch)
    backend = ClaudeCliSessionBackend()
    _open(backend, AgentClaude("", ""))

    with pytest.raises(SessionHostError) as raised:
        if method == "compact":
            backend.compact("")
        else:
            backend.fork()

    assert raised.value.operation == method


def test_claude_rejects_an_unsuccessful_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = CaptureTransport([CaptureOutcome('{"is_error": true}')])
    transport.install(monkeypatch)
    backend = ClaudeCliSessionBackend()
    _open(backend, AgentClaude("", ""))

    with pytest.raises(SessionHostError) as raised:
        backend.compact("")

    assert raised.value.operation == "compact"


def test_pi_first_invocation_failure_does_not_repeat_creation_flags_until_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome(returncode=1, stderr="failed"),
            CaptureOutcome("resumed"),
            CaptureOutcome("reset"),
        ]
    )
    transport.install(monkeypatch)
    agent = AgentPi("p", "m", "t")
    backend = PiCliSessionBackend()
    _open(backend, agent, name="named")

    with pytest.raises(SessionAskError):
        backend.ask(SessionAskRequest("first"))
    backend.ask(SessionAskRequest("retry"))
    backend.reset()
    backend.ask(SessionAskRequest("after reset"))

    original_id = transport.calls[0][0][3]
    _file_prompt_argv(
        transport.calls[0][0],
        [
            "pi",
            "-p",
            "--session-id",
            original_id,
            "--name",
            "named",
            "--provider",
            "p",
            "--model",
            "m",
            "--thinking",
            "t",
        ],
    )
    _file_prompt_argv(
        transport.calls[1][0],
        [
            "pi",
            "-p",
            "--session-id",
            original_id,
            "--provider",
            "p",
            "--model",
            "m",
            "--thinking",
            "t",
        ],
    )
    reset_id = transport.calls[2][0][3]
    assert reset_id != original_id
    _file_prompt_argv(
        transport.calls[2][0],
        [
            "pi",
            "-p",
            "--session-id",
            reset_id,
            "--name",
            "named",
            "--provider",
            "p",
            "--model",
            "m",
            "--thinking",
            "t",
        ],
    )


def test_codex_rejects_a_name_when_opening() -> None:
    with pytest.raises(SessionHostError) as raised:
        _open(CodexCliSessionBackend(), AgentCodex("", ""), name="named")

    assert raised.value.operation == "set-name"


def test_codex_does_not_retry_a_failed_first_invocation_as_a_new_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome(returncode=1, stderr="failed"),
            CaptureOutcome(
                '{"type":"thread.started","thread_id":"reset"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}'
            ),
        ]
    )
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("", ""))

    with pytest.raises(SessionAskError):
        backend.ask(SessionAskRequest("first"))
    with pytest.raises(SessionHostError) as raised:
        backend.ask(SessionAskRequest("retry"))
    backend.reset()
    assert backend.ask(SessionAskRequest("after reset")).content == "answer"

    assert raised.value.operation == SessionOperation.ASK
    assert [argv for argv, _ in transport.calls] == [
        ["codex", "exec", "--json", "-"],
        ["codex", "exec", "--json", "-"],
    ]


@pytest.mark.parametrize(
    ("backend", "agent", "operation", "args"),
    [
        (ClaudeCliSessionBackend, AgentClaude("", ""), "set_name", ("name",)),
        (ClaudeCliSessionBackend, AgentClaude("", ""), "stats", ()),
        (CodexCliSessionBackend, AgentCodex("", ""), "compact", ("",)),
        (CodexCliSessionBackend, AgentCodex("", ""), "fork", ()),
        (CodexCliSessionBackend, AgentCodex("", ""), "set_name", ("name",)),
        (CodexCliSessionBackend, AgentCodex("", ""), "stats", ()),
        (PiCliSessionBackend, AgentPi("", "", ""), "compact", ("",)),
        (PiCliSessionBackend, AgentPi("", "", ""), "set_name", ("name",)),
        (PiCliSessionBackend, AgentPi("", "", ""), "stats", ()),
    ],
)
def test_native_cli_sessions_reject_unsupported_operations(
    backend: Callable[[], object], agent: object, operation: str, args: tuple[str, ...]
) -> None:
    instance = backend()
    _open(instance, agent)

    with pytest.raises(SessionHostError) as raised:
        getattr(instance, operation)(*args)

    assert raised.value.operation == operation.replace("_", "-")


@pytest.mark.parametrize(
    ("backend", "agent"),
    [
        (ClaudeCliSessionBackend, AgentClaude("", "")),
        (CodexCliSessionBackend, AgentCodex("", "")),
        (PiCliSessionBackend, AgentPi("", "", "")),
    ],
)
def test_native_cli_sessions_close_and_reject_future_asks(
    backend: Callable[[], object], agent: object
) -> None:
    instance = backend()
    _open(instance, agent)
    instance.close()

    with pytest.raises(SessionHostError):
        instance.ask(SessionAskRequest("again"))


@pytest.mark.parametrize(
    ("backend", "agent"),
    [
        (ClaudeCliSessionBackend, AgentCodex("", "")),
        (CodexCliSessionBackend, AgentPi("", "", "")),
        (PiCliSessionBackend, AgentClaude("", "")),
    ],
)
def test_native_cli_sessions_reject_an_agent_for_another_backend(
    backend: Callable[[], object], agent: object
) -> None:
    with pytest.raises(SessionHostError) as raised:
        _open(backend(), agent)
    assert raised.value.operation == "open"
