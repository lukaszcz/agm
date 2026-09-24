"""Protocol validation and resource-boundary tests for Pi RPC sessions."""

from __future__ import annotations

import io
import itertools
import os
import queue
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

import pytest

from agm.agent.session import (
    SessionAskError,
    SessionAskRequest,
    SessionHostError,
    SessionOpenRequest,
    rpc,
)
from agm.agent.spec import AgentClaude, AgentPi
from tests._agl_helpers import unavailable_sandbox_context
from tests.test_agent_rpc import RpcStub, open_backend

_PI = AgentPi(provider="provider", model="model", thinking="think")

#: Fake children are given process groups above this floor so the autouse
#: fixture below can recognize — and never actually signal — them.
_FAKE_PROCESS_GROUP_FLOOR = 1 << 30
_fake_process_groups = itertools.count(_FAKE_PROCESS_GROUP_FLOOR)


@pytest.fixture(autouse=True)
def killed_groups(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the teardown of fake children instead of signalling a real group."""
    killed: list[int] = []
    send_signal = os.killpg

    def killpg(group: int, sig: int) -> None:
        if group < _FAKE_PROCESS_GROUP_FLOOR:
            send_signal(group, sig)
            return
        killed.append(group)
        raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", killpg)
    return killed


def _child(process: object) -> rpc._RpcChild:
    """Build an ``_RpcChild`` around a fake *process* with a plausible argv.

    ``kill_process_group`` always targets ``process.pid`` now (the dead
    ``pgid`` override was removed), so every fake process is stamped
    unconditionally with a synthetic, always-above-
    ``_FAKE_PROCESS_GROUP_FLOOR`` id here -- unique per call, overwriting any
    pid the fake already carries -- so the ``killed_groups`` fixture always
    recognizes and intercepts it instead of a real ``os.killpg`` ever
    reaching a real process group. A bare ``object()`` (used only by tests
    that never reach termination) is left untouched: it rejects arbitrary
    attributes.
    """
    popen = cast(subprocess.Popen[bytes], process)
    try:
        popen.pid = next(_fake_process_groups)
    except AttributeError:
        pass
    return rpc._RpcChild(popen, _PI, ["pi", "--mode", "rpc"])


def _reap_child_after_send(monkeypatch: pytest.MonkeyPatch) -> None:
    """Close the exit race deterministically: settle each send, then reap the child."""
    send = rpc.PiRpcSessionBackend._send

    def send_then_await_exit(
        self: rpc.PiRpcSessionBackend,
        operation: rpc._RpcOperation,
        payload: dict[str, object],
        *,
        wait_for_settled: bool = False,
    ) -> tuple[dict[str, object], list[str]]:
        result = send(self, operation, payload, wait_for_settled=wait_for_settled)
        child = self._child
        assert child is not None
        child.process.wait(timeout=5)
        return result

    monkeypatch.setattr(rpc.PiRpcSessionBackend, "_send", send_then_await_exit)


@pytest.mark.parametrize(
    "event",
    [
        {"raw": "not json"},
        {"raw": "[]"},
        {"raw": "{}"},
        {"raw": ""},
        {"raw": '{"type":"response","id":1,"command":"prompt","success":true}'},
        {"raw": '{"type":"response","id":"$id","command":"prompt","success":"true"}'},
        {
            "raw": (
                '{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":1}}'
            )
        },
        {"raw": '{"type":"message_update"}'},
        {"raw": '{"type":"response","id":"$id","command":"prompt","success":true,"x":NaN}'},
        {"raw": '{"type":"response","id":"$id","id":"again","command":"prompt","success":true}'},
        {
            "raw": (
                '{"type":"message_update","assistantMessageEvent":'
                '{"type":"text_delta","delta":"\\ud800"}}'
            )
        },
    ],
)
def test_malformed_protocol_records_kill_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, event: dict[str, str]
) -> None:
    RpcStub(tmp_path, monkeypatch, {"prompt": [event]})
    backend = open_backend()
    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest("hello"))
    assert raised.value.cause == "protocol_failure"
    with pytest.raises(SessionHostError):
        backend.compact("")
    backend.close()


def test_post_ack_terminal_failure_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(
        tmp_path,
        monkeypatch,
        {
            "prompt": [
                {"id": "$id", "type": "response", "command": "prompt", "success": True},
                {
                    "type": "message_end",
                    "message": {"stopReason": "error", "errorMessage": "provider failed"},
                },
                {"type": "agent_settled"},
            ]
        },
    )
    backend = open_backend()
    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest("hello"))
    assert raised.value.stderr_tail == "provider failed"
    backend.compact("")
    backend.close()


def test_retry_discards_failed_attempt_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(
        tmp_path,
        monkeypatch,
        {
            "prompt": [
                {"id": "$id", "type": "response", "command": "prompt", "success": True},
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": "stale"},
                },
                {"type": "auto_retry_start", "willRetry": True},
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": "new"},
                },
                {"type": "agent_settled"},
            ]
        },
    )
    backend = open_backend()

    assert backend.ask(SessionAskRequest("hello")).content == "new"
    backend.close()


def test_message_end_output_replaces_streaming_deltas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(
        tmp_path,
        monkeypatch,
        {
            "prompt": [
                {"id": "$id", "type": "response", "command": "prompt", "success": True},
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": "original"},
                },
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "hidden"},
                            {"type": "text", "text": "replacement"},
                            {"type": "text", "text": " output"},
                        ],
                    },
                },
                {"type": "agent_settled"},
            ]
        },
    )
    backend = open_backend()

    assert backend.ask(SessionAskRequest("hello")).content == "replacement output"
    backend.close()


def test_retrying_terminal_error_is_cleared_by_eventual_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(
        tmp_path,
        monkeypatch,
        {
            "prompt": [
                {"id": "$id", "type": "response", "command": "prompt", "success": True},
                {"type": "message_end", "message": {"stopReason": "error"}},
                {"type": "compaction_end", "result": None, "aborted": False, "willRetry": True},
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": "recovered"},
                },
                {"type": "agent_settled"},
            ]
        },
    )
    backend = open_backend()
    assert backend.ask(SessionAskRequest("hello")).content == "recovered"
    backend.close()


def test_malformed_terminal_event_kills_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(
        tmp_path,
        monkeypatch,
        {
            "prompt": [
                {"id": "$id", "type": "response", "command": "prompt", "success": True},
                {"type": "agent_end", "messages": None},
            ]
        },
    )
    backend = open_backend()
    with pytest.raises(SessionAskError):
        backend.ask(SessionAskRequest("hello"))
    with pytest.raises(SessionHostError):
        backend.compact("")
    backend.close()


def test_protocol_error_is_retained_alongside_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(tmp_path, monkeypatch, {"prompt": [{"raw": "[]"}]})
    backend = open_backend()
    child = backend._child
    assert child is not None
    child.stderr.append(b"Pi diagnostic")
    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest("hello"))
    assert "Pi diagnostic" in raised.value.stderr_tail
    assert "Pi RPC JSONL event was not an object" in raised.value.stderr_tail
    backend.close()


def test_bounded_records_output_and_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    too_large = "x" * (rpc._MAX_JSONL_RECORD_BYTES + 1)
    RpcStub(tmp_path, monkeypatch, {"prompt": [{"raw": too_large}]})
    backend = open_backend()
    with pytest.raises(SessionAskError):
        backend.ask(SessionAskRequest("hello"))
    backend.close()

    class Unwritable:
        stdin: io.BufferedWriter | None = None

    child = _child(Unwritable())
    child.stderr.append(b"x" * (rpc._MAX_STDERR_BYTES + 1))
    assert len(child.stderr.data) == rpc._MAX_STDERR_BYTES
    with pytest.raises(BrokenPipeError):
        rpc._write_command(child, {"type": "prompt"})


def test_stderr_reassembles_a_multibyte_character_split_across_chunks() -> None:
    """Bytes stay undecoded until read: a character split across two reader
    chunks must not turn into two replacement characters."""
    child = _child(object())
    child.stderr.append("café".encode()[:-1])  # split inside the trailing "é"
    child.stderr.append("café".encode()[-1:])
    assert rpc._stderr(child, "fallback") == "café"


def test_stderr_drops_an_orphaned_leading_continuation_byte() -> None:
    """Cutting the bounded tail to its byte bound can strand a continuation
    byte at the front; that artifact of the cut is dropped, not decoded."""
    child = _child(object())
    child.stderr.append(b"a" + "é".encode() + b"n" * (rpc._MAX_STDERR_BYTES - 1))
    assert rpc._stderr(child, "fallback") == "n" * (rpc._MAX_STDERR_BYTES - 1)


def test_stderr_rejects_an_untruncated_leading_continuation_byte() -> None:
    child = _child(object())
    child.stderr.append("é".encode()[-1:] + b"diagnostic")
    assert rpc._stderr(child, "fallback") == "fallback; stderr is not valid UTF-8 at byte 0"


def test_stderr_undecodable_tail_gives_empty_text_and_an_offset_note() -> None:
    child = _child(object())
    child.stderr.append(b"\xff\xfe")
    assert rpc._stderr(child, "fallback") == "fallback; stderr is not valid UTF-8 at byte 0"


def test_terminate_closes_stdin() -> None:
    class Stdin:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Process:
        def __init__(self) -> None:
            self.stdin = Stdin()

        def poll(self) -> int:
            return 0

    process = Process()
    rpc._terminate(_child(process))
    assert process.stdin.closed


def test_terminate_ignores_stdin_close_failure() -> None:
    class Stdin:
        def close(self) -> None:
            raise OSError("closed")

    class Process:
        stdin = Stdin()

        def poll(self) -> int:
            return 0

    rpc._terminate(_child(Process()))


def test_helpers_and_spawn_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        rpc._event_text_delta(
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "x"},
            }
        )
        == "x"
    )
    assert rpc._event_text_delta({"type": "agent_start"}) is None
    assert rpc._extension_ui_cancellation({"type": "agent_start"}) is None
    assert (
        rpc._extension_ui_cancellation(
            {"type": "extension_ui_request", "id": "notice", "method": "notify"}
        )
        is None
    )
    assert rpc._extension_ui_cancellation(
        {"type": "extension_ui_request", "id": "dialog", "method": "confirm"}
    ) == {"type": "extension_ui_response", "id": "dialog", "cancelled": True}
    with pytest.raises(rpc._RpcProtocolError):
        rpc._extension_ui_cancellation({"type": "extension_ui_request", "method": "confirm"})
    assert (
        rpc._terminal_prompt_failure(
            {"type": "auto_retry_end", "success": False, "finalError": "bad"}
        )
        == "bad"
    )
    assert (
        rpc._terminal_prompt_failure(
            {"type": "compaction_end", "result": None, "aborted": False, "willRetry": False}
        )
        == "Pi RPC compaction failed"
    )
    assert (
        rpc._terminal_prompt_failure({"type": "agent_end", "messages": [{"stopReason": "aborted"}]})
        == "Pi RPC agent aborted"
    )
    with pytest.raises(rpc._RpcProtocolError):
        rpc._terminal_prompt_failure({"type": "agent_end", "messages": [None]})
    assert rpc._session_id({}) is None
    assert rpc._session_id({"data": {"sessionId": "id"}}) == "id"
    with pytest.raises(rpc._RpcProtocolError):
        rpc._required_session_id({}, "fork")
    with pytest.raises(rpc._RpcProtocolError):
        rpc._require_not_cancelled({}, "fork")
    with pytest.raises(SessionHostError):
        rpc._require_not_cancelled({"data": {"cancelled": True}}, "fork")
    with pytest.raises(rpc._RpcProtocolError):
        rpc._validate_response({"type": "response", "id": "", "command": "x", "success": True})
    assert rpc._finite_decimal("1.5") is not None
    assert rpc._finite_decimal(True) is None
    assert rpc._finite_decimal("NaN") is None

    class Process:
        stdin = None

        def __init__(self) -> None:
            self.calls = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            raise ProcessLookupError

        def wait(self, timeout: float | None = None) -> None:
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("pi", timeout or 0)

        def kill(self) -> None:
            raise ProcessLookupError

    rpc._terminate(_child(Process()))

    class NoPipes:
        pid = 999_999_999
        stdin = None
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    def no_pipes(*args: object, **kwargs: object) -> NoPipes:
        return NoPipes()

    monkeypatch.setattr(subprocess, "Popen", no_pipes)
    with pytest.raises(SessionHostError):
        rpc.PiRpcSessionBackend(get_sandbox_context=unavailable_sandbox_context).open(
            SessionOpenRequest(AgentPi("", "", ""), "rpc")
        )


def test_rpc_private_protocol_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = rpc.PiRpcSessionBackend(get_sandbox_context=unavailable_sandbox_context)
    with pytest.raises(SessionHostError):
        backend.open(SessionOpenRequest(AgentClaude("model", "high"), "rpc"))

    with pytest.raises(InvalidOperation):
        rpc._parse_json_float("1e999999999999999999999999")
    with pytest.raises(rpc._RpcProtocolError):
        rpc._validate_response({"id": "id", "command": "", "success": True})
    with pytest.raises(rpc._RpcProtocolError):
        rpc._validate_response({"id": "id", "command": "prompt", "success": False})
    with pytest.raises(rpc._RpcProtocolError):
        rpc._event_text_delta({"type": "message_update", "assistantMessageEvent": {}})
    assert (
        rpc._event_text_delta({"type": "message_update", "assistantMessageEvent": {"type": "tool"}})
        is None
    )
    assert rpc._event_assistant_text({"type": "message_end", "message": None}) is None
    assert (
        rpc._event_assistant_text(
            {"type": "message_end", "message": {"role": "user", "content": []}}
        )
        is None
    )
    malformed_messages = (
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": [None]},
        {"role": "assistant", "content": [{"type": "text", "text": None}]},
    )
    for message in malformed_messages:
        with pytest.raises(rpc._RpcProtocolError):
            rpc._event_assistant_text({"type": "message_end", "message": message})
    with pytest.raises(rpc._RpcProtocolError):
        rpc._response_streaming_state({"data": {}})
    with pytest.raises(rpc._RpcProtocolError):
        rpc._terminal_prompt_failure({"type": "message_end", "message": None})
    assert (
        rpc._terminal_prompt_failure({"type": "message_end", "message": {"stopReason": "stop"}})
        is None
    )
    assert rpc._session_id({"data": {"sessionId": ""}}) is None

    malformed_stats: tuple[dict[str, object], ...] = (
        {},
        {"data": {}},
        {"data": {"tokens": {}}},
        {"data": {"tokens": {"input": -1, "output": 0}, "cost": 0}},
        {"data": {"tokens": {"input": 0, "output": 0}, "cost": -1}},
        {"data": {"tokens": {"input": 0, "output": 0}, "cost": 0, "contextUsage": []}},
    )
    for response in malformed_stats:
        with pytest.raises(rpc._RpcProtocolError):
            rpc._stats_from_response(response)
    zero_tokens: dict[str, object] = {"input": 0, "output": 0}
    no_context: dict[str, object] = {"tokens": zero_tokens, "cost": 0}
    assert rpc._stats_from_response({"data": no_context}).context_percent == Decimal("0")
    empty_context: dict[str, object] = {}
    missing_percent: dict[str, object] = {
        "tokens": zero_tokens,
        "cost": 0,
        "contextUsage": empty_context,
    }
    assert rpc._stats_from_response({"data": missing_percent}).context_percent == Decimal("0")
    assert rpc._finite_decimal("invalid") is None

    class RunningProcess:
        stdin = cast(io.BufferedWriter, io.BytesIO())

        def poll(self) -> None:
            return None

    child = _child(RunningProcess())
    child.stdout.put(b"\r\n")
    with pytest.raises(rpc._RpcProtocolError):
        backend._next_line(child)
    child.stdout.put(b"\xff\n")
    with pytest.raises(rpc._RpcProtocolError):
        backend._next_line(child)
    child.stdout.put(b"x" * (rpc._MAX_JSONL_RECORD_BYTES + 1) + b"\n")
    with pytest.raises(rpc._RpcProtocolError):
        backend._next_line(child)
    child.stdout_buffer.clear()
    backend._idle_timeout = 0
    with pytest.raises(rpc._RpcIdleTimeout):
        backend._next_line(child)

    child.stdout_buffer[:] = b"x" * (rpc._MAX_JSONL_RECORD_BYTES + 1) + b"\n"
    with pytest.raises(rpc._RpcProtocolError):
        backend._next_line(child)

    class FullQueue:
        def put(self, value: bytes | None, timeout: float) -> None:
            del value, timeout
            full.stopped.set()
            raise queue.Full

    full = _child(RunningProcess())
    full.stdout = cast(queue.Queue[bytes | None], FullQueue())
    rpc._queue_stdout(full, b"blocked")

    def fail_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        raise OSError("unavailable")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    with pytest.raises(SessionHostError):
        rpc.PiRpcSessionBackend(get_sandbox_context=unavailable_sandbox_context).open(
            SessionOpenRequest(AgentPi("", "", ""), "rpc")
        )

    with pytest.raises(ValueError):
        rpc._parse_json_float("Infinity")
    assert (
        rpc._terminal_prompt_failure(
            {"type": "compaction_end", "result": "kept", "willRetry": False}
        )
        is None
    )
    with pytest.raises(rpc._RpcProtocolError):
        rpc._terminal_prompt_failure({"type": "agent_end", "messages": None})


def test_open_rejects_an_already_live_child_and_dead_child_is_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(tmp_path, monkeypatch)
    backend = open_backend()
    with pytest.raises(SessionHostError):
        backend.open(SessionOpenRequest(AgentPi("provider", "model", "high"), "again"))
    backend.close()

    class DeadProcess:
        stdin = cast(io.BufferedWriter, io.BytesIO())

        def poll(self) -> int:
            return 1

    backend._child = _child(DeadProcess())
    with pytest.raises(SessionHostError):
        backend.compact("")
    assert backend._child is None
    backend._kill_dead_child(_child(DeadProcess()))


@pytest.mark.parametrize(
    "actions",
    [
        {"prompt": [{"id": "wrong", "type": "response", "command": "prompt", "success": True}]},
        {"compact": "exit"},
    ],
)
def test_unexpected_ack_and_non_prompt_transport_errors_close_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, actions: dict[str, object]
) -> None:
    RpcStub(tmp_path, monkeypatch, actions)
    backend = open_backend()
    with pytest.raises(SessionAskError if "prompt" in actions else SessionHostError) as raised:
        backend.ask(SessionAskRequest("hello")) if "prompt" in actions else backend.compact("")
    if "prompt" in actions:
        assert cast(SessionAskError, raised.value).cause == "protocol_failure"
    with pytest.raises(SessionHostError):
        backend.compact("")
    backend.close()


@pytest.mark.parametrize("action", ["new_session", "get_state"])
def test_malformed_operation_payload_kills_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    RpcStub(
        tmp_path,
        monkeypatch,
        {action: [{"id": "$id", "type": "response", "command": "$command", "success": True}]},
    )
    backend = open_backend()
    with pytest.raises(SessionHostError):
        backend.reset() if action == "new_session" else backend.fork()
    with pytest.raises(SessionHostError):
        backend.compact("")
    backend.close()


def test_malformed_stats_payload_kills_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(tmp_path, monkeypatch, stats={"tokens": {}})
    backend = open_backend()
    with pytest.raises(SessionHostError):
        backend.stats()
    with pytest.raises(SessionHostError):
        backend.compact("")
    backend.close()


def test_prompt_output_limit_and_write_failure_close_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(
        tmp_path,
        monkeypatch,
        {
            "prompt": [
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": "too long"},
                }
            ]
        },
    )
    monkeypatch.setattr(rpc, "_MAX_PROMPT_CHARS", 1)
    backend = open_backend()
    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest("hello"))
    assert raised.value.cause == "protocol_failure"
    backend.close()

    second = tmp_path / "second"
    second.mkdir()
    RpcStub(second, monkeypatch)
    backend = open_backend()

    def fail_write(
        child: rpc._RpcChild, command: dict[str, object], idle_timeout: float | None
    ) -> None:
        del child, command, idle_timeout
        raise OSError("closed")

    monkeypatch.setattr(rpc, "_write_command", fail_write)
    with pytest.raises(SessionAskError):
        backend.ask(SessionAskRequest("hello"))
    backend.close()


def test_fork_after_clone_failure_retains_replacement_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        stdin = cast(io.BufferedWriter, io.BytesIO())

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> None:
            del timeout

    backend = rpc.PiRpcSessionBackend(get_sandbox_context=unavailable_sandbox_context)
    source = _child(Process())
    replacement = _child(Process())
    backend._child = source
    source_states = iter([{"data": {"sessionId": "parent"}}, {}])

    def send(
        self: rpc.PiRpcSessionBackend, operation: str, payload: dict[str, object], **kwargs: object
    ) -> tuple[dict[str, object], list[str]]:
        del payload, kwargs
        if self is not backend:
            return {"data": {"sessionId": "parent"}}, []
        if operation == "get_state":
            return next(source_states), []
        return {"data": {"cancelled": False}}, []

    def spawn(agent: AgentPi, command: list[str], operation: str) -> rpc._RpcChild:
        del agent, command, operation
        return replacement

    monkeypatch.setattr(rpc.PiRpcSessionBackend, "_send", send)
    monkeypatch.setattr(backend, "_spawn", spawn)
    with pytest.raises(SessionHostError):
        backend.fork()
    assert backend._child is replacement
    backend.close()


def test_fork_replacement_readiness_failure_leaves_source_unchanged(
    monkeypatch: pytest.MonkeyPatch, killed_groups: list[int]
) -> None:
    class Process:
        stdin = cast(io.BufferedWriter, io.BytesIO())

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> None:
            del timeout

    backend = rpc.PiRpcSessionBackend(get_sandbox_context=unavailable_sandbox_context)
    source = _child(Process())
    replacement = _child(Process())
    backend._child = source

    def send(
        self: rpc.PiRpcSessionBackend, operation: str, payload: dict[str, object], **kwargs: object
    ) -> tuple[dict[str, object], list[str]]:
        del operation, payload, kwargs
        session_id = "parent" if self is backend else "wrong"
        return {"data": {"sessionId": session_id}}, []

    def spawn(agent: AgentPi, command: list[str], operation: str) -> rpc._RpcChild:
        del agent, command, operation
        return replacement

    monkeypatch.setattr(rpc.PiRpcSessionBackend, "_send", send)
    monkeypatch.setattr(backend, "_spawn", spawn)
    with pytest.raises(SessionHostError):
        backend.fork()
    assert backend._child is source
    assert set(killed_groups) == {replacement.process.pid}
    backend.close()


def test_fork_rejects_a_child_with_the_parent_session_id(
    monkeypatch: pytest.MonkeyPatch, killed_groups: list[int]
) -> None:
    class Process:
        stdin = cast(io.BufferedWriter, io.BytesIO())

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> None:
            del timeout

    backend = rpc.PiRpcSessionBackend(get_sandbox_context=unavailable_sandbox_context)
    source = _child(Process())
    replacement = _child(Process())
    backend._child = source
    calls: list[tuple[object, str]] = []

    def send(
        self: rpc.PiRpcSessionBackend, operation: str, payload: dict[str, object], **kwargs: object
    ) -> tuple[dict[str, object], list[str]]:
        del payload, kwargs
        calls.append((self, operation))
        if self is backend and operation == "clone":
            return {"data": {"cancelled": False}}, []
        return {"data": {"sessionId": "parent"}}, []

    def spawn(agent: AgentPi, command: list[str], operation: str) -> rpc._RpcChild:
        del agent, command, operation
        return replacement

    monkeypatch.setattr(rpc.PiRpcSessionBackend, "_send", send)
    monkeypatch.setattr(backend, "_spawn", spawn)
    with pytest.raises(SessionHostError):
        backend.fork()
    assert [operation for _, operation in calls] == ["get_state", "get_state", "clone", "get_state"]
    assert calls[1][0] is not backend
    assert backend._child is replacement
    assert set(killed_groups) == {source.process.pid}
    backend.close()


def test_fork_spawn_failure_does_not_move_the_live_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(tmp_path, monkeypatch)
    backend = open_backend()
    original = backend._child
    assert original is not None

    def fail_spawn(agent: AgentPi, command: list[str], operation: str) -> rpc._RpcChild:
        del agent, command
        raise SessionHostError("no", operation)

    monkeypatch.setattr(backend, "_spawn", fail_spawn)
    with pytest.raises(SessionHostError):
        backend.fork()
    assert backend._child is original
    backend.close()


def test_answer_survives_a_child_that_exits_before_the_response_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fully settled answer is returned even when Pi exits right after settling."""
    RpcStub(
        tmp_path,
        monkeypatch,
        {
            "get_state": [
                {
                    "id": "$id",
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "root", "isStreaming": False},
                },
                {"exit": True},
            ]
        },
    )
    backend = open_backend()
    _reap_child_after_send(monkeypatch)

    response = backend.ask(SessionAskRequest("hello"))

    assert response.content == "answer"
    assert response.call_info is not None
    assert response.call_info.exit_code == 0
    backend.close()


@pytest.mark.parametrize("child_exits", [False, True])
def test_malformed_payload_reports_the_protocol_violation_not_the_child_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, child_exits: bool
) -> None:
    """A malformed payload is reported as such even once Pi has already exited."""
    events: list[object] = [
        {
            "id": "$id",
            "type": "response",
            "command": "$command",
            "success": True,
            "data": {"tokens": {}},
        }
    ]
    if child_exits:
        events.append({"exit": True})
    RpcStub(tmp_path, monkeypatch, {"get_session_stats": events})
    backend = open_backend()
    if child_exits:
        _reap_child_after_send(monkeypatch)

    with pytest.raises(SessionHostError) as raised:
        backend.stats()

    assert isinstance(raised.value.__cause__, rpc._RpcProtocolError)
    assert backend._child is None
    backend.close()


def test_malformed_payload_is_reported_when_no_child_remains() -> None:
    """A closed session still reports why the payload was rejected."""
    backend = rpc.PiRpcSessionBackend(get_sandbox_context=unavailable_sandbox_context)

    with pytest.raises(SessionHostError) as raised:
        backend._parse_operation_response({}, "get_session_stats", rpc._required_session_id)

    assert isinstance(raised.value.__cause__, rpc._RpcProtocolError)
