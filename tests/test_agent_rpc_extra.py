"""Protocol validation and resource-boundary tests for Pi RPC sessions."""

from __future__ import annotations

import io
import itertools
import os
import queue
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import IO, cast

import pytest

from agm.agent.session import (
    SessionAskError,
    SessionAskRequest,
    SessionHostError,
    rpc,
)
from agm.agent.spec import AgentPi
from agm.agent.transport import AgentOutputPhase
from agm.sandbox.request import SandboxLimits
from tests._agl_helpers import (
    session_sandbox_context,
    unavailable_sandbox_context,
    write_sandbox_home,
)
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
        output_callback: rpc.AgentOutputCallback | None = None,
    ) -> tuple[dict[str, object], list[str]]:
        result = send(
            self,
            operation,
            payload,
            wait_for_settled=wait_for_settled,
            output_callback=output_callback,
        )
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
        {
            "raw": (
                '{"type":"response","id":"$id","command":"prompt","success":true,'
                '"x":1e99999999999999999999}'
            )
        },
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

    child = _child(object())
    child.stderr.append(b"x" * (rpc._MAX_STDERR_BYTES + 1))
    assert len(child.stderr.data) == rpc._MAX_STDERR_BYTES


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

    class Process:
        stdin = io.BytesIO()

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


def test_sandboxed_spawn_follows_the_process_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A sandboxed Pi RPC child starts in the process's current directory
    (e.g. after `os::chdir`), not the directory the sandbox context was built in.
    """
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")
    home = tmp_path / "home"
    write_sandbox_home(home)
    moved = tmp_path / "moved"
    moved.mkdir()
    monkeypatch.chdir(moved)
    spawned: dict[str, object] = {}

    def fake_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        spawned.update(kwargs)
        raise OSError("stop")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    with pytest.raises(SessionHostError):
        rpc.PiRpcSessionBackend.open(
            AgentPi("", "", ""),
            get_sandbox_context=session_sandbox_context(home),
            env={},
            sandbox=SandboxLimits(),
        )
    assert spawned["cwd"] == Path.cwd()


def test_sandboxed_spawn_with_removed_working_directory_is_a_session_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")
    home = tmp_path / "home"
    write_sandbox_home(home)
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    with pytest.raises(SessionHostError):
        rpc.PiRpcSessionBackend.open(
            AgentPi("", "", ""),
            get_sandbox_context=session_sandbox_context(home),
            env={},
            sandbox=SandboxLimits(),
        )


def test_rpc_private_protocol_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = rpc.PiRpcSessionBackend(
        _child(object()), idle_timeout=None, get_sandbox_context=unavailable_sandbox_context, env={}
    )

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
        rpc.PiRpcSessionBackend.open(
            AgentPi("", "", ""), get_sandbox_context=unavailable_sandbox_context, env={}
        )

    assert (
        rpc._terminal_prompt_failure(
            {"type": "compaction_end", "result": "kept", "willRetry": False}
        )
        is None
    )
    with pytest.raises(rpc._RpcProtocolError):
        rpc._terminal_prompt_failure({"type": "agent_end", "messages": None})


def test_pi_output_helpers_handle_empty_and_closed_output_channels() -> None:
    output: list[tuple[str, str]] = []

    def callback(phase: AgentOutputPhase, text: str, **_metadata: object) -> None:
        output.append((phase, text))

    assistant_progress = ["pending"]
    tool_output_lengths: dict[str, str] = {}
    rpc._emit_pi_progress(
        {"type": "message_end", "message": None},
        delta=None,
        assistant_progress=assistant_progress,
        tool_output_lengths=tool_output_lengths,
        callback=callback,
    )
    rpc._emit_pi_progress(
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "toolUse"}},
        delta=None,
        assistant_progress=assistant_progress,
        tool_output_lengths=tool_output_lengths,
        callback=callback,
    )
    rpc._emit_pi_progress(
        {"type": "tool_execution_start"},
        delta=None,
        assistant_progress=assistant_progress,
        tool_output_lengths=tool_output_lengths,
        callback=callback,
    )
    rpc._emit_pi_progress(
        {"type": "tool_execution_update", "partialResult": {"content": []}},
        delta=None,
        assistant_progress=assistant_progress,
        tool_output_lengths=tool_output_lengths,
        callback=callback,
    )
    rpc._emit_pi_progress(
        {
            "type": "tool_execution_update",
            "toolCallId": "empty",
            "partialResult": {"content": []},
        },
        delta=None,
        assistant_progress=assistant_progress,
        tool_output_lengths=tool_output_lengths,
        callback=callback,
    )
    assert assistant_progress == []
    assert output == []
    assert rpc._rpc_result_text(None) == ""
    assert rpc._rpc_result_text({"content": None}) == ""

    child = _child(object())

    def closed_channel(_phase: str, _text: str, **_metadata: object) -> None:
        raise BrokenPipeError

    child.output_callback = closed_channel
    rpc._capture_stderr(child, b"diagnostic")

    assert child.stderr.data == b"diagnostic"


class _StderrProcess:
    def __init__(self, stderr: IO[bytes] | None) -> None:
        self.stderr = stderr


def test_finish_stderr_delivery_drains_pending_pipe_bytes() -> None:
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb", buffering=0)
    child = _child(_StderrProcess(stream))
    output: list[tuple[str, str]] = []
    child.output_callback = lambda phase, text, **_metadata: output.append((phase, text))
    os.write(write_fd, b"pending diagnostic")
    os.close(write_fd)

    try:
        rpc._finish_stderr_delivery(child)
    finally:
        stream.close()

    assert output == [("stderr", "pending diagnostic")]
    assert child.stderr.data == b"pending diagnostic"
    assert child.output_callback is None


def test_finish_stderr_delivery_handles_missing_or_closed_streams() -> None:
    for stream in (None, io.BytesIO()):
        child = _child(_StderrProcess(stream))
        child.output_callback = lambda _phase, _text, **_metadata: None

        rpc._finish_stderr_delivery(child)

        assert child.output_callback is None


@pytest.mark.parametrize("failure", [OSError("closed"), ValueError("closed")])
def test_stderr_reader_stops_when_select_fails(
    monkeypatch: pytest.MonkeyPatch, failure: OSError | ValueError
) -> None:
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb", buffering=0)
    child = _child(_StderrProcess(stream))

    def fail_select(*_args: object) -> tuple[list[int], list[int], list[int]]:
        raise failure

    monkeypatch.setattr(rpc.select, "select", fail_select)
    reader = rpc._start_stderr_reader(child, stream)
    reader.join(timeout=5)
    os.close(write_fd)

    assert not reader.is_alive()
    assert child.stderr_reader_done.is_set()


def test_stderr_reader_stops_when_pipe_read_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb", buffering=0)
    child = _child(_StderrProcess(stream))
    os.write(write_fd, b"ready")

    def fail_read(_descriptor: int, _size: int) -> bytes:
        raise OSError("closed")

    monkeypatch.setattr(rpc.os, "read", fail_read)
    reader = rpc._start_stderr_reader(child, stream)
    reader.join(timeout=5)
    os.close(write_fd)

    assert not reader.is_alive()
    assert child.stderr_reader_done.is_set()


def test_stderr_reader_closes_a_stopped_child_stream() -> None:
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb", buffering=0)
    child = _child(_StderrProcess(stream))
    child.stopped.set()

    reader = rpc._start_stderr_reader(child, stream)
    reader.join(timeout=5)
    os.close(write_fd)

    assert not reader.is_alive()
    assert child.stderr_reader_done.is_set()


def test_dead_child_is_cleared(killed_groups: list[int]) -> None:
    class DeadProcess:
        stdin = cast(io.BufferedWriter, io.BytesIO())

        def poll(self) -> int:
            return 1

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 1

    dead = _child(DeadProcess())
    backend = rpc.PiRpcSessionBackend(
        dead, idle_timeout=None, get_sandbox_context=unavailable_sandbox_context, env={}
    )
    with pytest.raises(SessionHostError):
        backend.compact("")
    assert backend._child is None
    assert killed_groups == [dead.process.pid]


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

    source = _child(Process())
    replacement = _child(Process())
    backend = rpc.PiRpcSessionBackend(
        source, idle_timeout=None, get_sandbox_context=unavailable_sandbox_context, env={}
    )
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

    def spawn(
        agent: AgentPi, command: list[str], operation: str, **_kwargs: object
    ) -> rpc._RpcChild:
        del agent, command, operation
        return replacement

    monkeypatch.setattr(rpc.PiRpcSessionBackend, "_send", send)
    monkeypatch.setattr(rpc, "_spawn", spawn)
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

    source = _child(Process())
    replacement = _child(Process())
    backend = rpc.PiRpcSessionBackend(
        source, idle_timeout=None, get_sandbox_context=unavailable_sandbox_context, env={}
    )

    def send(
        self: rpc.PiRpcSessionBackend, operation: str, payload: dict[str, object], **kwargs: object
    ) -> tuple[dict[str, object], list[str]]:
        del operation, payload, kwargs
        session_id = "parent" if self is backend else "wrong"
        return {"data": {"sessionId": session_id}}, []

    def spawn(
        agent: AgentPi, command: list[str], operation: str, **_kwargs: object
    ) -> rpc._RpcChild:
        del agent, command, operation
        return replacement

    monkeypatch.setattr(rpc.PiRpcSessionBackend, "_send", send)
    monkeypatch.setattr(rpc, "_spawn", spawn)
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

    source = _child(Process())
    replacement = _child(Process())
    backend = rpc.PiRpcSessionBackend(
        source, idle_timeout=None, get_sandbox_context=unavailable_sandbox_context, env={}
    )
    calls: list[tuple[object, str]] = []

    def send(
        self: rpc.PiRpcSessionBackend, operation: str, payload: dict[str, object], **kwargs: object
    ) -> tuple[dict[str, object], list[str]]:
        del payload, kwargs
        calls.append((self, operation))
        if self is backend and operation == "clone":
            return {"data": {"cancelled": False}}, []
        return {"data": {"sessionId": "parent"}}, []

    def spawn(
        agent: AgentPi, command: list[str], operation: str, **_kwargs: object
    ) -> rpc._RpcChild:
        del agent, command, operation
        return replacement

    monkeypatch.setattr(rpc.PiRpcSessionBackend, "_send", send)
    monkeypatch.setattr(rpc, "_spawn", spawn)
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

    def fail_spawn(
        agent: AgentPi, command: list[str], operation: str, **_kwargs: object
    ) -> rpc._RpcChild:
        del agent, command
        raise SessionHostError("no", operation)

    monkeypatch.setattr(rpc, "_spawn", fail_spawn)
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


def _stats_with(cost: object, percent: object) -> dict[str, object]:
    return {
        "data": {
            "tokens": {"input": 1, "output": 2},
            "cost": cost,
            "contextUsage": {"percent": percent},
        }
    }


@pytest.mark.parametrize("value", [True, "NaN", "invalid", "1e1000000", Decimal("1e-1000030")])
@pytest.mark.parametrize("field", ["cost", "percent"])
def test_stats_reject_malformed_or_out_of_range_numbers(field: str, value: object) -> None:
    """A stats number that is not a finite decimal within the AgL range is malformed."""
    response = _stats_with(value, 1) if field == "cost" else _stats_with(1, value)
    with pytest.raises(rpc._RpcProtocolError):
        rpc._stats_from_response(response)


def test_stats_accept_textual_and_decimal_numbers() -> None:
    stats = rpc._stats_from_response(_stats_with("1.5", Decimal("2.5")))
    assert (stats.cost, stats.context_percent) == (Decimal("1.5"), Decimal("2.5"))


def test_malformed_payload_is_reported_when_no_child_remains() -> None:
    """A closed session still reports why the payload was rejected."""
    backend = rpc.PiRpcSessionBackend(
        _child(object()), idle_timeout=None, get_sandbox_context=unavailable_sandbox_context, env={}
    )
    backend._child = None

    with pytest.raises(SessionHostError) as raised:
        backend._parse_operation_response({}, "get_session_stats", rpc._required_session_id)

    assert isinstance(raised.value.__cause__, rpc._RpcProtocolError)
