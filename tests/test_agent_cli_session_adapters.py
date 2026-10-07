"""Unit tests for native CLI-continuation agent session adapters."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    AgentCommandSessionBackend,
    ClaudeCliSessionBackend,
    CodexCliSessionBackend,
    PiCliSessionBackend,
    open_cli_session,
)
from agm.agent.session.protocol import BackendSettings
from agm.agent.spec import (
    AGENT_SPECS,
    AgentClaude,
    AgentCodex,
    AgentCommand,
    AgentPi,
    AgentSpec,
    PermissionMode,
    SessionTransport,
    payload_fields,
)
from agm.agl.semantics.type_table import BUILTIN_PRELUDE_TYPE_DEFS, create_seeded_type_table
from agm.core.process import CapturedOutput, ProcessCaptureResult
from agm.sandbox.request import PreparedSandboxCommand, SandboxLimits
from tests._agl_helpers import (
    session_sandbox_context,
    unavailable_sandbox_context,
    write_sandbox_home,
)


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
        self.envs: list[dict[str, str] | None] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def run(argv: list[str], **kwargs: object) -> ProcessCaptureResult:
            stdin_text = kwargs.get("stdin_text")
            self.calls.append((argv, stdin_text if isinstance(stdin_text, str) else None))
            env = kwargs.get("env")
            self.envs.append(env if isinstance(env, dict) else None)
            outcome = self.outcomes.pop(0)
            stdout_callback = kwargs.get("stdout_callback")
            if callable(stdout_callback) and outcome.stdout:
                stdout_callback(outcome.stdout)
            stderr_callback = kwargs.get("stderr_callback")
            if callable(stderr_callback) and outcome.stderr:
                stderr_callback(outcome.stderr)
            return ProcessCaptureResult(
                returncode=outcome.returncode,
                stdout=CapturedOutput(data=outcome.stdout.encode(), truncated=outcome.timed_out),
                stderr=CapturedOutput(data=outcome.stderr.encode(), truncated=outcome.timed_out),
                elapsed=0.1,
                timed_out=outcome.timed_out,
                spawn_error=outcome.spawn_error,
            )

        monkeypatch.setattr("agm.agent.runner.run_capture_result", run)


def _open(
    agent: AgentSpec,
    *,
    name: str = "",
    single_prompt: bool = False,
    permission_mode: PermissionMode = PermissionMode.NONE,
    sandbox: SandboxLimits | None = None,
    env: dict[str, str] | None = None,
    get_sandbox_context=unavailable_sandbox_context,
) -> Any:
    return open_cli_session(
        SessionOpenRequest(
            agent=agent,
            transport=SessionTransport.CLI,
            name=name,
            single_prompt=single_prompt,
            permission_mode=permission_mode,
            sandbox=sandbox,
            env=env or {},
        ),
        idle_timeout=None,
        get_sandbox_context=get_sandbox_context,
    )


@pytest.mark.parametrize("single_prompt", [False, True])
@pytest.mark.parametrize("echo", [False, True])
def test_codex_captures_large_events_from_nonblocking_stdout(
    tmp_path: Path, single_prompt: bool, echo: bool
) -> None:
    executable = tmp_path / "codex"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "sys.stdin.read()\n"
        "os.set_blocking(1, False)\n"
        "answer = 'x' * 2_000_000\n"
        "if '--json' in sys.argv:\n"
        "    output = json.dumps({'type': 'thread.started', 'thread_id': 'thread'}) + '\\n'\n"
        "    output += json.dumps({'type': 'item.completed', 'item': "
        "{'type': 'agent_message', 'text': answer}}) + '\\n'\n"
        "else:\n"
        "    output = answer\n"
        "data = output.encode()\n"
        "if os.write(1, data) != len(data):\n"
        "    sys.stderr.write('stdout backpressure\\n')\n"
        "    sys.exit(101)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    backend = _open(
        AgentCodex("", ""),
        get_sandbox_context=unavailable_sandbox_context,
        single_prompt=single_prompt,
        env={"PATH": str(tmp_path)},
    )

    events: list[tuple[str, str]] = []
    request = SessionAskRequest(
        "question",
        output_callback=(lambda phase, text, **_metadata: events.append((phase, text)))
        if echo
        else None,
    )

    assert backend.ask(request).content == "x" * 2_000_000
    if not single_prompt:
        assert backend.ask(request).content == "x" * 2_000_000
    assert not any(phase == "stderr" for phase, _ in events)
    backend.close()


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

    assert set(declared) == set(expected) == set(AGENT_SPECS)
    for variant, (fields, spec, backend, transport) in expected.items():
        assert tuple(name for name, _ in declared[variant]) == fields
        assert payload_fields(spec) == fields
        assert AGENT_SPECS[variant] is spec
        assert type(_open(spec(*("x",) * len(fields)), single_prompt=True)) is backend
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
    backend = _open(agent, name="named")

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
    backend = _open(AgentClaude("m", "t"))
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
    parent = _open(AgentClaude("m", "t"), name="named")

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
    backend = _open(AgentClaude("m", "t"), name="named")

    backend.compact("instructions")
    backend.ask(SessionAskRequest("question"))

    assert len(transport.calls) == 1
    assert "--session-id" in transport.calls[0][0]
    assert "--resume" not in transport.calls[0][0]
    assert "named" in transport.calls[0][0]


def test_claude_echo_stream_decodes_final_response_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "\n".join(
        [
            '{"type":"stream_event","event":{"type":"content_block_start",'
            '"content_block":{"type":"tool_use","name":"Read"}}}',
            '{"type":"stream_event","event":{"type":"content_block_delta",'
            '"delta":{"type":"text_delta","text":"working"}}}',
            '{"type":"assistant","message":{"stop_reason":"tool_use"}}',
            '{"type":"stream_event","event":{"type":"content_block_delta",'
            '"delta":{"type":"text_delta","text":"answer"}}}',
            '{"type":"assistant","message":{"stop_reason":"end_turn"}}',
            '{"type":"result","result":"decoded final"}',
        ]
    )
    transport = CaptureTransport([CaptureOutcome(output, stderr="diagnostic\n")])
    transport.install(monkeypatch)
    backend = _open(AgentClaude("m", "t"), get_sandbox_context=unavailable_sandbox_context)

    output_chunks: list[tuple[str, str]] = []

    response = backend.ask(
        SessionAskRequest(
            "question",
            output_callback=lambda phase, text, **_metadata: output_chunks.append((phase, text)),
        )
    )

    assert response.content == "decoded final"
    assert output_chunks == [
        ("progress", "[Read]\n"),
        ("progress", "working"),
        ("stderr", "diagnostic\n"),
    ]
    command = transport.calls[0][0]
    assert command[:2] == ["claude", "-p"]
    assert command[command.index("--output-format") + 1] == "stream-json"
    assert "--include-partial-messages" in command
    assert "--verbose" in command


def test_claude_echo_stream_rejects_an_undecodable_final_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport([CaptureOutcome("not json")])
    transport.install(monkeypatch)
    backend = _open(AgentClaude("m", "t"), get_sandbox_context=unavailable_sandbox_context)

    with pytest.raises(SessionAskError) as raised:
        backend.ask(
            SessionAskRequest("question", output_callback=lambda _phase, _text, **_metadata: None)
        )

    assert raised.value.cause == "protocol_failure"


def test_codex_echo_stream_decodes_response_and_echoes_command_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "\n".join(
        [
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"item.completed","item":{"type":"agent_message",'
            '"text":"I will inspect the files."}}',
            '{"type":"item.started","item":{"type":"command_execution","command":"ls"}}',
            '{"type":"item.completed","item":{"type":"command_execution",'
            '"aggregated_output":"file.txt\\n"}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"final"}}',
        ]
    )
    resumed_output = "\n".join(
        [
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"item.completed","item":{"type":"agent_message",'
            '"text":"I will check one more thing."}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"resumed"}}',
        ]
    )
    transport = CaptureTransport(
        [CaptureOutcome(output, stderr="codex log\n"), CaptureOutcome(resumed_output)]
    )
    transport.install(monkeypatch)
    backend = _open(AgentCodex("m", "t"), get_sandbox_context=unavailable_sandbox_context)

    output_chunks: list[tuple[str, str]] = []

    response = backend.ask(
        SessionAskRequest(
            "question",
            output_callback=lambda phase, text, **_metadata: output_chunks.append((phase, text)),
        )
    )

    assert response.content == "final"
    assert output_chunks == [
        ("progress", "I will inspect the files."),
        ("progress", "$ ls\n"),
        ("progress", "file.txt\n"),
        ("stderr", "codex log\n"),
    ]
    assert transport.calls[0][0][:4] == ["codex", "exec", "--json", "--model"]
    resumed = backend.ask(
        SessionAskRequest("again", output_callback=lambda _phase, _text, **_metadata: None)
    )
    assert resumed.content == "resumed"
    assert transport.calls[1][0][:5] == ["codex", "exec", "resume", "thread-1", "--json"]


def test_codex_single_prompt_echo_decodes_jsonl_final_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "\n".join(
        [
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"item.started","item":{"type":"command_execution","command":"pwd"}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}',
        ]
    )
    transport = CaptureTransport([CaptureOutcome(output)])
    transport.install(monkeypatch)
    backend = _open(
        AgentCodex("m", "t"), get_sandbox_context=unavailable_sandbox_context, single_prompt=True
    )

    output_chunks: list[tuple[str, str]] = []

    response = backend.ask(
        SessionAskRequest(
            "question",
            output_callback=lambda phase, text, **_metadata: output_chunks.append((phase, text)),
        )
    )

    assert response.content == "answer"
    assert output_chunks == [("progress", "$ pwd\n")]
    assert transport.calls[0][0][:4] == ["codex", "exec", "--json", "--model"]


def test_codex_single_prompt_echo_rejects_malformed_jsonl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport([CaptureOutcome("not json")])
    transport.install(monkeypatch)
    backend = _open(
        AgentCodex("m", "t"), get_sandbox_context=unavailable_sandbox_context, single_prompt=True
    )

    with pytest.raises(SessionAskError) as raised:
        backend.ask(
            SessionAskRequest("question", output_callback=lambda _phase, _text, **_metadata: None)
        )

    assert raised.value.cause == "protocol_failure"


def test_codex_single_prompt_echo_reports_turn_failure_as_agent_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "\n".join(
        (
            '{"type":"thread.started","thread_id":"thread-1"}',
            '{"type":"turn.failed","error":{"message":"upstream rate limit"}}',
        )
    )
    transport = CaptureTransport([CaptureOutcome(output)])
    transport.install(monkeypatch)
    backend = _open(
        AgentCodex("m", "t"), get_sandbox_context=unavailable_sandbox_context, single_prompt=True
    )

    with pytest.raises(SessionAskError) as raised:
        backend.ask(
            SessionAskRequest("question", output_callback=lambda _phase, _text, **_metadata: None)
        )

    assert raised.value.cause == "nonzero_exit"
    assert "upstream rate limit" in raised.value.stderr_tail


def test_pi_forks_immediately_after_open_then_child_starts_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport([CaptureOutcome("child"), CaptureOutcome("parent")])
    transport.install(monkeypatch)
    parent = _open(AgentPi("p", "m", "t"), name="named")

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
    parent = _open(AgentPi("p", "m", "t"))

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
    ("agent", "name_flag"),
    [
        (AgentClaude("m", "t"), "-n"),
        (AgentPi("p", "m", "t"), "--name"),
    ],
)
def test_session_id_cli_backends_reuse_ids_and_names_after_reset(
    monkeypatch: pytest.MonkeyPatch,
    agent: AgentClaude | AgentPi,
    name_flag: str,
) -> None:
    transport = CaptureTransport(
        [CaptureOutcome("first"), CaptureOutcome("second"), CaptureOutcome("reset")]
    )
    transport.install(monkeypatch)
    session = _open(agent, name="named")

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
    backend = _open(AgentCodex("m", "t"), single_prompt=True)

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
    backend = _open(AgentCodex("m", "t"))

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
    backend = _open(agent)

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
    backend = _open(AgentCodex("m", "t"))

    assert backend.ask(SessionAskRequest("hello")).content == "\U0001f600"


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_codex_reply_preserves_unicode_jsonl_separators(
    monkeypatch: pytest.MonkeyPatch, separator: str
) -> None:
    expected = f"left{separator}right"
    output = (
        '{"type":"thread.started","thread_id":"first"}\n'
        '{"type":"item.completed","item":{"type":"agent_message","text":"'
        f"{expected}"
        '"}}'
    )
    transport = CaptureTransport([CaptureOutcome(output)])
    transport.install(monkeypatch)
    backend = _open(AgentCodex("m", "t"), get_sandbox_context=unavailable_sandbox_context)

    assert backend.ask(SessionAskRequest("hello")).content == expected


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
    backend = _open(AgentCodex("", ""))

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
    backend = _open(AgentCodex("", ""))

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
    backend = _open(AgentCodex("model", "high"))

    backend.reset()
    assert transport.calls == []

    assert backend.ask(SessionAskRequest("create")).content == "answer"
    assert len(transport.calls) == 1


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
    backend = _open(agent, name="named")

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
    service = SessionService(
        lambda request: open_cli_session(
            request, idle_timeout=None, get_sandbox_context=unavailable_sandbox_context
        )
    )
    handle = service.open(AgentClaude("", ""), SessionTransport.CLI, env={})
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
    backend = _open(AgentClaude("", ""))
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
    backend = _open(AgentClaude("", ""))
    backend.ask(SessionAskRequest("start"))

    with pytest.raises(SessionHostError) as raised:
        backend.compact("")

    assert raised.value.operation == "compact"


@pytest.mark.parametrize(
    ("agent", "name_flag"),
    [
        (AgentClaude("m", "t"), "-n"),
        (AgentPi("p", "m", "t"), "--name"),
    ],
)
def test_spawn_failure_retries_cli_session_creation(
    monkeypatch: pytest.MonkeyPatch,
    agent: AgentClaude | AgentPi,
    name_flag: str,
) -> None:
    transport = CaptureTransport([CaptureOutcome(spawn_error="missing"), CaptureOutcome("answer")])
    transport.install(monkeypatch)
    session = _open(agent, name="named")

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
    backend = _open(agent, name="named")

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
    "agent",
    [AgentCommand("runner --session %{SESSION_ID}"), AgentCodex("", "")],
)
def test_backends_rejecting_a_name_report_the_open_operation(agent: AgentSpec) -> None:
    with pytest.raises(SessionHostError) as raised:
        _open(agent, name="named")

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
    backend = _open(AgentCodex("", ""))

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
    backend = _open(AgentCodex("", ""))

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
    ("agent", "supported"),
    [
        (AgentCommand("runner --session %{SESSION_ID}"), set()),
        (AgentClaude("", ""), {"compact", "fork"}),
        (AgentCodex("", ""), set()),
        (AgentPi("", "", ""), {"fork"}),
    ],
)
def test_native_cli_sessions_expose_only_their_native_operations(
    agent: AgentSpec, supported: set[str]
) -> None:
    operations = _open(agent).operations

    assert {
        name
        for name in ("compact", "fork", "set_name", "stats")
        if getattr(operations, name) is not None
    } == supported


def test_claude_single_prompt_session_uses_the_standard_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport([CaptureOutcome("answer")])
    transport.install(monkeypatch)
    backend = _open(AgentClaude("m", "t"), single_prompt=True)

    assert backend.ask(SessionAskRequest("prompt")).content == "answer"
    (argv, _stdin) = transport.calls[0]
    assert "--session-id" not in argv
    _file_prompt_argv(argv, AgentClaude("m", "t").argv())


def test_pi_single_prompt_session_uses_the_standard_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = CaptureTransport([CaptureOutcome("answer")])
    transport.install(monkeypatch)
    backend = _open(AgentPi("p", "m", "t"), single_prompt=True)

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
    backend = _open(AgentCodex("m", "t"))

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
    backend = _open(AgentCodex("", ""))

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
    backend = _open(AgentCodex("", ""))

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
    backend = _open(AgentCodex("", ""))

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
    backend = _open(AgentCodex("m", "t"))

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
    backend = _open(AgentCodex("", ""))

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
    backend = _open(AgentCodex("", ""))

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


def _sandbox_wrapped_argv_prefix(argv: list[str], home: Path) -> None:
    """Assert *argv* is wrapped by the systemd-run/srt chain rooted at *home*."""
    assert argv[:4] == ["systemd-run", "--user", "--scope", "-q"]
    srt_index = argv.index("srt")
    assert argv[srt_index + 1] == "--settings"
    assert argv[srt_index + 2] == str(home / ".agm" / "sandbox" / "default.json")
    assert argv[srt_index + 3] == "--"


def test_agent_command_session_wraps_each_prompt_under_sandbox_mode_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The session's own ``Sandbox`` mode -- fixed at open -- wraps the argv for
    every prompt it sends, and each prepared sandbox command is closed right
    after that prompt -- never held open across prompts."""
    home = tmp_path / "home"
    write_sandbox_home(home)
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")
    transport = CaptureTransport([CaptureOutcome("first"), CaptureOutcome("second")])
    transport.install(monkeypatch)

    closed_before_close: list[bool] = []
    original_close = PreparedSandboxCommand.close

    def spy_close(self: PreparedSandboxCommand) -> None:
        closed_before_close.append(self._closed)
        original_close(self)

    monkeypatch.setattr(PreparedSandboxCommand, "close", spy_close)

    limits = SandboxLimits()
    backend = _open(
        AgentCommand("cat %{SESSION_ID}"),
        get_sandbox_context=session_sandbox_context(home),
        permission_mode=PermissionMode.UNRESTRICTED,
        sandbox=limits,
    )

    first = backend.ask(SessionAskRequest("first"))
    second = backend.ask(SessionAskRequest("second"))

    assert first.content == "first"
    assert second.content == "second"
    assert len(transport.calls) == 2
    for argv, _stdin in transport.calls:
        _sandbox_wrapped_argv_prefix(argv, home)
    # Each prepared command started unclosed and was closed exactly once,
    # independently of the other prompt.
    assert closed_before_close == [False, False]


def test_agent_command_session_ask_under_disabled_reproduces_the_unwrapped_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Disabled`` (the default) never consults sandboxing and reproduces the
    plain, unwrapped argv a command session always sent before sandboxing existed."""
    transport = CaptureTransport([CaptureOutcome("answer")])
    transport.install(monkeypatch)
    backend = _open(
        AgentCommand("cat %{SESSION_ID}"), get_sandbox_context=unavailable_sandbox_context
    )

    response = backend.ask(SessionAskRequest("hello"))

    assert response.content == "answer"
    [(argv, _stdin)] = transport.calls
    session = backend._session
    assert session is not None
    _file_prompt_argv(argv, ["cat", session.session_id])


def test_claude_session_open_sandbox_mode_wraps_open_compact_and_fork_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Claude session's mode is fixed at open and every native lifecycle
    prompt it sends -- compaction, forking -- reuses it, not only ``ask``."""
    home = tmp_path / "home"
    write_sandbox_home(home)
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")
    transport = CaptureTransport(
        [
            CaptureOutcome("first"),
            CaptureOutcome('{"is_error": false}'),
            CaptureOutcome('{"session_id": "child"}'),
        ]
    )
    transport.install(monkeypatch)

    backend = ClaudeCliSessionBackend.open(
        AgentClaude("m", "t"),
        BackendSettings(
            get_sandbox_context=session_sandbox_context(home),
            permission_mode=PermissionMode.UNRESTRICTED,
            sandbox=SandboxLimits(),
            env={"FIXED": "at-open"},
        ),
    )

    backend.ask(SessionAskRequest("first"))
    backend.compact("")
    child = backend.fork()

    assert len(transport.calls) == 3
    for argv, _stdin in transport.calls:
        _sandbox_wrapped_argv_prefix(argv, home)
    # Every native call this session made -- the initial prompt, compaction,
    # and the fork itself -- ran under the same environment fixed at open
    # (the sandbox backend may add its own entries on top, but never drops
    # or changes the fixed one).
    assert transport.envs[0] == transport.envs[1] == transport.envs[2]
    assert transport.envs[0] is not None and transport.envs[0]["FIXED"] == "at-open"
    assert isinstance(child, ClaudeCliSessionBackend)
    assert child._settings == backend._settings


def test_sandboxed_codex_disables_the_daemon_on_creation_resume_and_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    write_sandbox_home(home)
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")
    created = CaptureOutcome(
        '{"type":"thread.started","thread_id":"first"}\n'
        '{"type":"item.completed","item":{"type":"agent_message","text":"answer"}}'
    )
    transport = CaptureTransport([created, CaptureOutcome("resumed"), created])
    transport.install(monkeypatch)
    backend = _open(
        AgentCodex("m", "t"),
        get_sandbox_context=session_sandbox_context(home),
        permission_mode=PermissionMode.UNRESTRICTED,
        sandbox=SandboxLimits(),
    )

    assert backend.ask(SessionAskRequest("first")).content == "answer"
    assert backend.ask(SessionAskRequest("second")).content == "resumed"
    backend.reset()
    assert backend.ask(SessionAskRequest("after reset")).content == "answer"

    for argv, _stdin in transport.calls:
        _sandbox_wrapped_argv_prefix(argv, home)
        assert argv[argv.index("codex") + 1] == "--no-daemon"
        assert argv.count("--no-daemon") == 1
