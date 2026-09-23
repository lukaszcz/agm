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
from agm.agent.spec import (
    AGENT_SPECS,
    AgentClaude,
    AgentCodex,
    AgentCommand,
    AgentPi,
    AgentSpec,
    payload_fields,
)
from agm.agl.semantics.type_table import BUILTIN_PRELUDE_TYPE_DEFS, create_seeded_type_table
from agm.core.process import CapturedOutput, ProcessCaptureResult


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
                stdout=CapturedOutput(data=outcome.stdout.encode(), truncated=outcome.timed_out),
                stderr=CapturedOutput(data=outcome.stderr.encode(), truncated=outcome.timed_out),
                elapsed=0.1,
                timed_out=outcome.timed_out,
                spawn_error=outcome.spawn_error,
            )

        monkeypatch.setattr("agm.agent.runner.run_capture_result", run)


def _open(backend: object, agent: object, *, name: str = "", single_prompt: bool = False) -> None:
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
    backend.open(
        SessionOpenRequest(agent=agent, transport="cli", name=name, single_prompt=single_prompt)
    )


def _file_prompt_argv(argv: list[str], command: list[str]) -> None:
    """Assert normal file delivery without pinning its temporary path."""
    assert argv[:-1] == command
    assert argv[-1].startswith("@")


def test_agent_variant_spec_backend_catalogs_are_in_lockstep() -> None:
    expected: dict[str, tuple[tuple[str, ...], type[AgentSpec], type[SessionBackend], str]] = {
        "AgentCommand": (("command",), AgentCommand, AgentCommandSessionBackend, "Cli"),
        "AgentClaude": (("model", "thinking"), AgentClaude, ClaudeCliSessionBackend, "Cli"),
        "AgentCodex": (("model", "thinking"), AgentCodex, CodexCliSessionBackend, "Cli"),
        "AgentPi": (("provider", "model", "thinking"), AgentPi, PiCliSessionBackend, "Rpc"),
    }
    table = create_seeded_type_table()
    declared = {
        member.name: tuple(table.record_fields(member).items())
        for member in BUILTIN_PRELUDE_TYPE_DEFS["Agent"].members
    }

    assert set(declared) == set(expected) == set(AGENT_SPECS) == set(CLI_SESSION_BACKENDS)
    for variant, (fields, spec, backend, transport) in expected.items():
        assert tuple(name for name, _ in declared[variant]) == fields
        assert payload_fields(spec) == fields
        assert AGENT_SPECS[variant] is spec
        assert CLI_SESSION_BACKENDS[variant] is backend
        assert spec.DEFAULT_SESSION_TRANSPORT == transport


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
        "",
    )
    assert isinstance(child, ClaudeCliSessionBackend)


def test_claude_fork_session_id_combines_a_surrogate_escape_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A high+low escape pair in the fork's ``session_id`` is one scalar character."""
    high_escape = "\\" + "ud83d"
    low_escape = "\\" + "ude00"
    transport = CaptureTransport(
        [
            CaptureOutcome("first"),
            CaptureOutcome(f'{{"session_id": "id-{high_escape}{low_escape}"}}'),
            CaptureOutcome("child answer"),
        ]
    )
    transport.install(monkeypatch)
    backend = ClaudeCliSessionBackend()
    _open(backend, AgentClaude("m", "t"))
    backend.ask(SessionAskRequest("first"))

    child = backend.fork()
    assert child.ask(SessionAskRequest("child")).content == "child answer"

    child_argv = transport.calls[2][0]
    assert child_argv[child_argv.index("--resume") + 1] == "id-\U0001f600"


def test_claude_forks_immediately_after_open_with_independent_child_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport([CaptureOutcome("child answer"), CaptureOutcome("parent answer")])
    transport.install(monkeypatch)
    parent = ClaudeCliSessionBackend()
    _open(parent, AgentClaude("m", "t"), name="named")

    child = parent.fork()
    assert child.ask(SessionAskRequest("child follow-up")).content == "child answer"
    assert parent.ask(SessionAskRequest("parent follow-up")).content == "parent answer"

    child_id = transport.calls[0][0][3]
    parent_id = transport.calls[1][0][3]
    assert child_id != parent_id
    _file_prompt_argv(
        transport.calls[0][0],
        [
            "claude",
            "-p",
            "--session-id",
            child_id,
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


def test_claude_compact_before_first_ask_is_deferred(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = CaptureTransport([CaptureOutcome("answer")])
    transport.install(monkeypatch)
    backend = ClaudeCliSessionBackend()
    _open(backend, AgentClaude("m", "t"), name="named")

    backend.compact("instructions")
    backend.ask(SessionAskRequest("question"))

    assert len(transport.calls) == 1
    assert "--session-id" in transport.calls[0][0]
    assert "--resume" not in transport.calls[0][0]
    assert "named" in transport.calls[0][0]


def test_pi_forks_immediately_after_open_then_child_starts_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport([CaptureOutcome("child"), CaptureOutcome("parent")])
    transport.install(monkeypatch)
    parent = PiCliSessionBackend()
    _open(parent, AgentPi("p", "m", "t"), name="named")

    child = parent.fork()
    child.ask(SessionAskRequest("child"))
    parent.ask(SessionAskRequest("parent"))

    child_id = transport.calls[0][0][3]
    parent_id = transport.calls[1][0][3]
    assert child_id != parent_id
    _file_prompt_argv(
        transport.calls[0][0],
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
        transport.calls[1][0],
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


def test_pi_forks_started_transcript_natively(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = CaptureTransport(
        [CaptureOutcome("parent"), CaptureOutcome("forked"), CaptureOutcome("child")]
    )
    transport.install(monkeypatch)
    parent = PiCliSessionBackend()
    _open(parent, AgentPi("p", "m", "t"))

    parent.ask(SessionAskRequest("start"))
    child = parent.fork()
    child.ask(SessionAskRequest("continue"))

    parent_id = transport.calls[0][0][3]
    fork_argv = transport.calls[1][0]
    child_id = fork_argv[fork_argv.index("--session-id") + 1]
    assert fork_argv[fork_argv.index("--fork") + 1] == parent_id
    assert transport.calls[1][1] == ""
    assert child_id in transport.calls[2][0]


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


def test_codex_single_prompt_session_uses_the_standard_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport([CaptureOutcome("answer")])
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("m", "t"), single_prompt=True)

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
                        '{"type":"item.started","item":{"type":"command_execution"}}',
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

    first = backend.ask(SessionAskRequest("first"))
    assert first.content == "first answer"
    assert first.metadata == {"elapsed": 0.1}
    assert first.call_info is not None
    assert first.call_info.argv[:3] == ["codex", "exec", "--json"]
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


def test_codex_reply_combines_a_surrogate_escape_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """A high+low escape pair in an event's ``text`` field is one scalar character."""
    high_escape = "\\" + "ud83d"
    low_escape = "\\" + "ude00"
    transport = CaptureTransport(
        [
            CaptureOutcome(
                "\n".join(
                    (
                        '{"type":"thread.started","thread_id":"first"}',
                        '{"type":"item.completed","item":'
                        f'{{"type":"agent_message","text":"{high_escape}{low_escape}"}}}}',
                        '{"type":"turn.completed"}',
                    )
                )
            ),
        ]
    )
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("m", "t"))

    assert backend.ask(SessionAskRequest("hello")).content == "\U0001f600"


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
        '{"type":"thread.started","thread_id":"first"}\n{"type":"item.completed","item":'
        '{"type":"agent_message","text":"\\ud800"}}',
    ],
)
def test_codex_rejects_malformed_or_incomplete_jsonl(
    monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    transport = CaptureTransport([CaptureOutcome(output)])
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("", ""))

    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest("first"))

    assert raised.value.cause == "protocol_failure"
    assert raised.value.call_info.argv[:3] == ["codex", "exec", "--json"]
    assert raised.value.stderr_tail


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
    transport = CaptureTransport(
        [CaptureOutcome("started"), CaptureOutcome(returncode=1, stderr="failed")]
    )
    transport.install(monkeypatch)
    backend = ClaudeCliSessionBackend()
    service = SessionService(lambda _agent, _transport: backend)
    handle = service.open(AgentClaude("", ""), "cli")
    service.ask(handle, SessionAskRequest("start"))

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
        ("compact", '{"is_error": false, "note": "\\ud800"}'),
        ("fork", "not json"),
        ("fork", "[]"),
        ("fork", "{}"),
        ("fork", '{"session_id": "\\ud800"}'),
    ],
)
def test_claude_lifecycle_protocol_errors_are_host_errors(
    monkeypatch: pytest.MonkeyPatch, method: str, output: str
) -> None:
    transport = CaptureTransport([CaptureOutcome("started"), CaptureOutcome(output)])
    transport.install(monkeypatch)
    backend = ClaudeCliSessionBackend()
    _open(backend, AgentClaude("", ""))
    backend.ask(SessionAskRequest("start"))

    with pytest.raises(SessionHostError) as raised:
        if method == "compact":
            backend.compact("")
        else:
            backend.fork()

    assert raised.value.operation == method


def test_claude_rejects_an_unsuccessful_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = CaptureTransport([CaptureOutcome("started"), CaptureOutcome('{"is_error": true}')])
    transport.install(monkeypatch)
    backend = ClaudeCliSessionBackend()
    _open(backend, AgentClaude("", ""))
    backend.ask(SessionAskRequest("start"))

    with pytest.raises(SessionHostError) as raised:
        backend.compact("")

    assert raised.value.operation == "compact"


@pytest.mark.parametrize(
    ("backend", "agent", "name_flag"),
    [
        (ClaudeCliSessionBackend, AgentClaude("m", "t"), "-n"),
        (PiCliSessionBackend, AgentPi("p", "m", "t"), "--name"),
    ],
)
def test_spawn_failure_retries_cli_session_creation(
    monkeypatch: pytest.MonkeyPatch,
    backend: Callable[[], ClaudeCliSessionBackend | PiCliSessionBackend],
    agent: AgentClaude | AgentPi,
    name_flag: str,
) -> None:
    transport = CaptureTransport([CaptureOutcome(spawn_error="missing"), CaptureOutcome("answer")])
    transport.install(monkeypatch)
    session = backend()
    _open(session, agent, name="named")

    with pytest.raises(SessionAskError):
        session.ask(SessionAskRequest("first"))
    session.ask(SessionAskRequest("retry"))

    assert name_flag in transport.calls[0][0]
    assert name_flag in transport.calls[1][0]


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


@pytest.mark.parametrize(
    ("backend", "agent"),
    [
        (AgentCommandSessionBackend, AgentCommand("runner --session %{SESSION_ID}")),
        (CodexCliSessionBackend, AgentCodex("", "")),
    ],
)
def test_backends_rejecting_a_name_report_the_open_operation(
    backend: Callable[[], object], agent: object
) -> None:
    with pytest.raises(SessionHostError) as raised:
        _open(backend(), agent, name="named")

    assert raised.value.operation == "open"


def test_codex_retries_thread_creation_after_spawn_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome(spawn_error="missing"),
            CaptureOutcome(
                '{"type":"thread.started","thread_id":"created"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}'
            ),
        ]
    )
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("", ""))

    with pytest.raises(SessionAskError):
        backend.ask(SessionAskRequest("first"))

    assert backend.ask(SessionAskRequest("retry")).content == "answer"
    assert [argv for argv, _ in transport.calls] == [
        ["codex", "exec", "--json", "-"],
        ["codex", "exec", "--json", "-"],
    ]


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


def test_claude_single_prompt_session_uses_the_standard_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport([CaptureOutcome("answer")])
    transport.install(monkeypatch)
    backend = ClaudeCliSessionBackend()
    _open(backend, AgentClaude("m", "t"), single_prompt=True)

    assert backend.ask(SessionAskRequest("prompt")).content == "answer"
    (argv, _stdin) = transport.calls[0]
    assert "--session-id" not in argv
    _file_prompt_argv(argv, AgentClaude("m", "t").argv())


def test_pi_single_prompt_session_uses_the_standard_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport([CaptureOutcome("answer")])
    transport.install(monkeypatch)
    backend = PiCliSessionBackend()
    _open(backend, AgentPi("p", "m", "t"), single_prompt=True)

    assert backend.ask(SessionAskRequest("prompt")).content == "answer"
    (argv, _stdin) = transport.calls[0]
    assert "--session-id" not in argv
    _file_prompt_argv(argv, AgentPi("p", "m", "t").argv())


def test_codex_turn_failure_reports_the_agent_error_not_a_protocol_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome(
                "\n".join(
                    (
                        '{"type":"thread.started","thread_id":"first"}',
                        '{"type":"turn.started"}',
                        '{"type":"turn.failed","error":{"message":"upstream rate limit"}}',
                    )
                )
            )
        ]
    )
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("m", "t"))

    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest("first"))

    assert raised.value.cause == "nonzero_exit"
    assert "upstream rate limit" in raised.value.stderr_tail
    assert raised.value.call_info.argv[:3] == ["codex", "exec", "--json"]


@pytest.mark.parametrize(
    "failure_event",
    [
        '{"type":"turn.failed"}',
        '{"type":"turn.failed","error":null}',
        '{"type":"turn.failed","error":"boom"}',
        '{"type":"turn.failed","error":{}}',
        '{"type":"turn.failed","error":{"message":42}}',
        '{"type":"turn.failed","error":{"message":""}}',
    ],
)
def test_codex_turn_failure_without_a_usable_message_still_fails_the_ask(
    monkeypatch: pytest.MonkeyPatch, failure_event: str
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome(
                "\n".join(('{"type":"thread.started","thread_id":"first"}', failure_event))
            )
        ]
    )
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("", ""))

    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest("first"))

    assert raised.value.cause == "nonzero_exit"
    assert raised.value.stderr_tail


def test_codex_ignores_unrecognized_protocol_events(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome(
                "\n".join(
                    (
                        '{"type":"thread.started","thread_id":"first"}',
                        '{"type":"turn.started"}',
                        '{"type":"turn.introduced_later","detail":{"kind":"unknown"}}',
                        '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}',
                        '{"type":"turn.completed"}',
                    )
                )
            )
        ]
    )
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("", ""))

    assert backend.ask(SessionAskRequest("first")).content == "answer"


def test_codex_tolerates_blank_lines_in_the_jsonl_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome(
                "\n".join(
                    (
                        "",
                        '{"type":"thread.started","thread_id":"first"}',
                        "   ",
                        '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}',
                        "\t",
                        "",
                    )
                )
            )
        ]
    )
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("", ""))

    assert backend.ask(SessionAskRequest("first")).content == "answer"


def test_codex_resumes_the_started_thread_after_a_failed_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome(
                "\n".join(
                    (
                        '{"type":"thread.started","thread_id":"first"}',
                        '{"type":"turn.started"}',
                        '{"type":"turn.failed","error":{"message":"upstream rate limit"}}',
                    )
                )
            ),
            CaptureOutcome("later answer"),
        ]
    )
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("m", "t"))

    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest("first"))
    assert raised.value.cause == "nonzero_exit"

    assert backend.ask(SessionAskRequest("later")).content == "later answer"
    assert [argv for argv, _ in transport.calls] == [
        ["codex", "exec", "--json", "--model", "m", "-c", "model_reasoning_effort=t", "-"],
        ["codex", "exec", "resume", "first", "--model", "m", "-c", "model_reasoning_effort=t", "-"],
    ]


def test_codex_starts_a_fresh_thread_after_a_turn_failure_without_a_thread_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome('{"type":"turn.failed","error":{"message":"model unavailable"}}'),
            CaptureOutcome(
                '{"type":"thread.started","thread_id":"second"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}'
            ),
        ]
    )
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("", ""))

    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest("first"))
    assert raised.value.cause == "nonzero_exit"

    assert backend.ask(SessionAskRequest("retry")).content == "answer"
    assert [argv for argv, _ in transport.calls] == [
        ["codex", "exec", "--json", "-"],
        ["codex", "exec", "--json", "-"],
    ]


def test_codex_keeps_a_malformed_stream_thread_unresumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport(
        [
            CaptureOutcome('{"type":"thread.started","thread_id":"first"}\n[]'),
            CaptureOutcome(
                '{"type":"thread.started","thread_id":"second"}\n'
                '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}'
            ),
        ]
    )
    transport.install(monkeypatch)
    backend = CodexCliSessionBackend()
    _open(backend, AgentCodex("", ""))

    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest("first"))
    assert raised.value.cause == "protocol_failure"
    with pytest.raises(SessionHostError) as host_error:
        backend.ask(SessionAskRequest("retry"))
    assert host_error.value.operation == SessionOperation.ASK

    backend.reset()
    assert backend.ask(SessionAskRequest("after reset")).content == "answer"
    assert [argv for argv, _ in transport.calls] == [
        ["codex", "exec", "--json", "-"],
        ["codex", "exec", "--json", "-"],
    ]
