"""Persistent JSONL RPC sessions for Pi."""

from __future__ import annotations

import codecs
import json
import os
import queue
import select
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import IO, Literal, Protocol, Self, TypeVar, cast
from uuid import uuid4

from agm.agent.session.protocol import (
    BackendSettings,
    SandboxFixture,
    SessionAskError,
    SessionAskRequest,
    SessionAskResponse,
    SessionBackend,
    SessionHostError,
    SessionOperation,
    SessionOperations,
    SessionStats,
)
from agm.agent.spec import AgentPi
from agm.agent.transport import (
    AgentCallInfo,
    AgentOutputCallback,
    AgentTransportFailureCause,
    stderr_tail,
)
from agm.core.process import CapturedOutput, stop_process
from agm.sandbox.backend import SandboxSettingsError, SandboxUnavailableError
from agm.sandbox.prepare import sandbox_run_for
from agm.sandbox.profile import profile_name
from agm.sandbox.request import PreparedSandboxCommand
from agm.util.decimal import decimal_in_range, parse_json_decimal
from agm.util.unicode import loads_exact_json

_RpcOperation = Literal[
    "prompt",
    "compact",
    "new_session",
    "clone",
    "set_session_name",
    "get_session_stats",
    "get_state",
]

_MAX_STDOUT_CHUNKS = 64
_MAX_STDERR_BYTES = 16_384
_MAX_PROMPT_CHARS = 4_194_304
_CONTEXT_PERCENT_UNAVAILABLE = Decimal("0")
_T = TypeVar("_T")


@dataclass(slots=True)
class _BoundedText:
    """Keep only the diagnostic byte tail produced by one child stream.

    Bytes stay undecoded until an error needs them (:func:`_stderr`):
    decoding each incoming chunk on its own, as it arrives, would turn a
    multi-byte character split across two chunks into two replacement
    characters instead of the one character it actually is.
    """

    data: bytes = b""
    truncated: bool = False

    def append(self, chunk: bytes) -> None:
        data = self.data + chunk
        if len(data) > _MAX_STDERR_BYTES:
            self.truncated = True
        self.data = data[-_MAX_STDERR_BYTES:]

    def captured(self) -> CapturedOutput:
        """The retained tail as a capture: its head, never its tail, was cut."""
        return CapturedOutput(data=self.data, truncated=False, head_truncated=self.truncated)


class _Lock(Protocol):
    """The lock interface used to synchronize stderr delivery."""

    def acquire(self) -> bool: ...

    def release(self) -> None: ...


@dataclass(slots=True)
class _RpcChild:
    """The process and asynchronously drained streams for one Pi session."""

    process: subprocess.Popen[bytes]
    agent: AgentPi
    command: list[str]
    prepared: PreparedSandboxCommand | None = None
    stdout_buffer: bytearray = field(default_factory=bytearray)
    stdout: queue.Queue[bytes | None] = field(
        default_factory=lambda: queue.Queue(maxsize=_MAX_STDOUT_CHUNKS)
    )
    stderr: _BoundedText = field(default_factory=_BoundedText)
    stderr_decoder: codecs.IncrementalDecoder = field(
        default_factory=lambda: codecs.getincrementaldecoder("utf-8")("replace")
    )
    stderr_lock: _Lock = field(default_factory=threading.Lock)
    stderr_reader_done: threading.Event = field(default_factory=threading.Event)
    output_callback: AgentOutputCallback | None = None
    readers: list[threading.Thread] = field(default_factory=list)
    stopped: threading.Event = field(default_factory=threading.Event)

    @property
    def stdin(self) -> IO[bytes]:
        """The command pipe, which :func:`_spawn` always opens."""
        return cast(IO[bytes], self.process.stdin)


class PiRpcSessionBackend(SandboxFixture):
    """Keep one Pi RPC process alive for the lifetime of a session backend.

    ``_child`` is ``None`` once the process was terminated: by ``close`` or
    after a transport failure.
    """

    def __init__(
        self,
        child: _RpcChild,
        settings: BackendSettings,
    ) -> None:
        super().__init__(settings)
        self._child: _RpcChild | None = child

    @classmethod
    def open(
        cls,
        agent: AgentPi,
        settings: BackendSettings,
        *,
        name: str = "",
    ) -> Self:
        """Start Pi in RPC mode using *agent*'s settings."""
        child = _spawn(
            agent,
            agent.rpc_argv(name=name, permission_mode=settings.permission_mode),
            "open",
            settings,
        )
        return cls(child, settings)

    continues_conversation = True

    @property
    def operations(self) -> SessionOperations:
        """Pi RPC implements every optional operation natively."""
        return SessionOperations(
            compact=self.compact, fork=self.fork, set_name=self.set_name, stats=self.stats
        )

    def ask(self, request: SessionAskRequest) -> SessionAskResponse:
        """Send a prompt and collect its text deltas through the settled event."""
        started = time.monotonic()
        # Resolve the child before sending: a process that exits once the answer is
        # settled must not discard that answer, so report its state without requiring
        # it to still be alive here.
        child = self._live_child("prompt")
        try:
            _, text = self._send(
                "prompt",
                {"message": request.prompt},
                wait_for_settled=True,
                output_callback=request.output_callback,
            )
        finally:
            _finish_stderr_delivery(child, drain=False)
        elapsed = time.monotonic() - started
        return SessionAskResponse(
            content="".join(text),
            metadata={"elapsed": elapsed},
            call_info=AgentCallInfo(
                argv=child.command.copy(),
                prompt_via_stdin=True,
                elapsed=elapsed,
                exit_code=child.process.poll(),
                sandboxed=child.prepared is not None,
                permission_mode=self._settings.permission_mode.value,
            ),
        )

    def compact(self, instructions: str) -> None:
        """Request native Pi compaction, optionally with custom instructions."""
        payload: dict[str, object] = {}
        if instructions:
            payload["customInstructions"] = instructions
        self._send("compact", payload)

    def reset(self) -> None:
        """Start a fresh conversation within the persistent Pi process."""
        response, _ = self._send("new_session", {})
        self._parse_operation_response(response, "new_session", _require_not_cancelled)

    def fork(self) -> SessionBackend:
        """Clone the active branch and atomically hand each process to one backend.

        Pi's ``fork`` command only accepts a previous user-message entry.  ``clone``
        is the RPC operation that snapshots the current active branch, regardless of
        whether its leaf is a user message or an assistant message.
        """
        parent_state, _ = self._send("get_state", {})
        parent_id = self._parse_operation_response(parent_state, "get_state", _required_session_id)
        source = self._live_child("clone")
        parent_command = source.agent.rpc_argv(
            session_id=parent_id, permission_mode=self._settings.permission_mode
        )
        replacement = _spawn(
            source.agent,
            parent_command,
            SessionOperation.FORK.value,
            self._settings,
        )
        replacement_backend = PiRpcSessionBackend(replacement, self._settings)
        try:
            replacement_state, _ = replacement_backend._send("get_state", {})
            replacement_backend._parse_operation_response(
                replacement_state,
                "get_state",
                lambda state, operation: _replacement_session_id(state, operation, parent_id),
            )
        except BaseException:
            replacement_backend.close()
            raise

        cloned = False
        try:
            response, _ = self._send("clone", {})
            self._parse_operation_response(response, "clone", _require_not_cancelled)
            cloned = True
            child_state, _ = self._send("get_state", {})
            self._parse_operation_response(
                child_state,
                "get_state",
                lambda state, operation: _forked_session_id(state, operation, parent_id),
            )
        except BaseException:
            if cloned:
                # The source now owns the clone, so retain the independently opened
                # parent rather than leaving this backend pointed at the wrong session.
                self._child = replacement
                _terminate(source)
            else:
                replacement_backend.close()
            raise

        self._child = replacement
        return PiRpcSessionBackend(source, self._settings)

    def set_name(self, name: str) -> None:
        """Set Pi's display name for the active session."""
        self._send("set_session_name", {"name": name})

    def stats(self) -> SessionStats:
        """Map Pi's session statistics payload into the backend-neutral shape."""
        response, _ = self._send("get_session_stats", {})
        return self._parse_operation_response(
            response, "get_session_stats", lambda payload, _operation: _stats_from_response(payload)
        )

    def close(self) -> None:
        """Terminate the child process; repeated closes are harmless."""
        child = self._child
        self._child = None
        if child is not None:
            _terminate(child)

    def _write(
        self,
        child: _RpcChild,
        command: dict[str, object],
        operation: _RpcOperation,
        started: float,
    ) -> None:
        """Write one command, mapping every stdin failure to the session model."""
        try:
            _write_command(child, command, self._settings.idle_timeout)
        except KeyboardInterrupt:
            self.close()
            raise
        except _RpcIdleTimeout as exc:
            self.close()
            self._raise_transport_or_host(
                operation, "Pi RPC stdin write timed out", started, exc, child
            )
        except (BrokenPipeError, OSError) as exc:
            self.close()
            self._raise_transport_or_host(operation, "Pi RPC stdin closed", started, exc, child)

    def _send(
        self,
        operation: _RpcOperation,
        payload: dict[str, object],
        *,
        wait_for_settled: bool = False,
        output_callback: AgentOutputCallback | None = None,
    ) -> tuple[dict[str, object], list[str]]:
        child = self._live_child(operation)
        request_id = str(uuid4())
        command = {"id": request_id, "type": operation, **payload}
        started = time.monotonic()
        child.output_callback = output_callback
        self._write(child, command, operation, started)

        text: list[str] = []
        text_length = 0
        assistant_progress: list[str] = []
        tool_output_lengths: dict[str, str] = {}
        tool_names: dict[str, str] = {}
        response: dict[str, object] | None = None
        state_request_id: str | None = None
        streaming: bool | None = None
        settled = False
        terminal_error: str | None = None
        while response is None or (
            wait_for_settled and (streaming is None or (streaming and not settled))
        ):
            streaming_state: bool | None = None
            try:
                event = self._next_event(child)
                ui_cancellation = _extension_ui_cancellation(event)
                if ui_cancellation is not None:
                    _write_command(child, ui_cancellation, self._settings.idle_timeout)
                delta = _event_text_delta(event)
                authoritative_text = _event_assistant_text(event)
                if output_callback is not None:
                    _emit_pi_progress(
                        event,
                        delta=delta,
                        assistant_progress=assistant_progress,
                        tool_output_lengths=tool_output_lengths,
                        tool_names=tool_names,
                        callback=output_callback,
                    )
                if event["type"] == "response":
                    _validate_response(event)
                    if event.get("id") == state_request_id and event.get("command") == "get_state":
                        streaming_state = _response_streaming_state(event)
                failure = _terminal_prompt_failure(event) if wait_for_settled else None
            except KeyboardInterrupt:
                self.close()
                raise
            except (BrokenPipeError, OSError) as exc:
                self.close()
                self._raise_transport_or_host(operation, "Pi RPC stdin closed", started, exc, child)
            except _RpcIdleTimeout as exc:
                self.close()
                self._raise_transport_or_host(operation, "Pi RPC idle timeout", started, exc, child)
            except _RpcProcessExited as exc:
                self.close()
                self._raise_transport_or_host(
                    operation, "Pi RPC process exited", started, exc, child
                )
            except _RpcProtocolError as exc:
                self.close()
                self._raise_transport_or_host(operation, str(exc), started, exc, child)

            event_type = event["type"]
            if event_type == "response":
                event_id = cast(str, event["id"])
                event_command = cast(str, event["command"])
                if event_id == request_id and event_command == operation:
                    if event["success"] is not True:
                        message = cast(str, event["error"])
                        if operation == "prompt":
                            self._raise_ask_error("nonzero_exit", message, started, child)
                        raise SessionHostError(message, _operation_name(operation))
                    response = event
                    if wait_for_settled:
                        state_request_id = str(uuid4())
                        self._write(
                            child,
                            {"id": state_request_id, "type": "get_state"},
                            operation,
                            started,
                        )
                elif event_id == state_request_id and event_command == "get_state":
                    streaming = cast(bool, streaming_state)
                else:
                    self.close()
                    self._raise_transport_or_host(
                        operation,
                        "Pi RPC returned an unexpected response",
                        started,
                        _RpcProtocolError("Pi RPC returned an unexpected response"),
                        child,
                    )
            if wait_for_settled:
                if event.get("willRetry") is True:
                    text.clear()
                    text_length = 0
                    terminal_error = None
                    assistant_progress.clear()
                    tool_output_lengths.clear()
                    tool_names.clear()
                else:
                    if authoritative_text is not None:
                        text = [authoritative_text]
                        text_length = len(authoritative_text)
                    elif delta is not None:
                        text.append(delta)
                        text_length += len(delta)
                    if text_length > _MAX_PROMPT_CHARS:
                        self.close()
                        self._raise_transport_or_host(
                            operation,
                            "Pi RPC prompt output exceeded the protocol limit",
                            started,
                            _RpcProtocolError("Pi RPC prompt output exceeded the protocol limit"),
                            child,
                        )
                    if failure is not None:
                        terminal_error = failure
                settled = settled or event_type == "agent_settled"
        _finish_stderr_delivery(child)
        if terminal_error is not None:
            self._raise_ask_error("nonzero_exit", terminal_error, started, child)
        return response, text

    def _parse_operation_response(
        self,
        response: dict[str, object],
        operation: _RpcOperation,
        parser: Callable[[dict[str, object], str], _T],
    ) -> _T:
        """Parse operation-specific data, killing a child that violates the protocol."""
        try:
            return parser(response, _operation_name(operation))
        except _RpcProtocolError as exc:
            # A process that has already exited must not mask the violation that
            # was diagnosed.
            self.close()
            raise SessionHostError(str(exc), _operation_name(operation)) from exc

    def _live_child(self, operation: _RpcOperation) -> _RpcChild:
        child = self._child
        if child is None:
            raise SessionHostError(
                "Pi RPC session process was terminated", _operation_name(operation)
            )
        if child.process.poll() is not None:
            self.close()
            raise SessionHostError("Pi RPC session process has exited", _operation_name(operation))
        return child

    def _next_event(self, child: _RpcChild) -> dict[str, object]:
        line = self._next_line(child)
        try:
            decoded: object = loads_exact_json(line)
        except ValueError as exc:
            raise _RpcProtocolError("Pi RPC returned malformed JSONL") from exc
        if not isinstance(decoded, dict):
            raise _RpcProtocolError("Pi RPC JSONL event was not an object")
        event = cast(dict[str, object], decoded)
        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            raise _RpcProtocolError("Pi RPC JSONL event had no type")
        return event

    def _next_line(self, child: _RpcChild) -> str:
        """Read an uncapped LF-delimited record; Pi events can include images and transcripts."""
        deadline = (
            None
            if self._settings.idle_timeout is None
            else time.monotonic() + self._settings.idle_timeout
        )
        while True:
            buffer = child.stdout_buffer
            newline = buffer.find(b"\n")
            if newline >= 0:
                line = bytes(buffer[:newline])
                del buffer[: newline + 1]
                if line.endswith(b"\r"):
                    line = line[:-1]
                if not line:
                    raise _RpcProtocolError("Pi RPC returned an empty JSONL record")
                try:
                    return line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise _RpcProtocolError("Pi RPC returned non-UTF-8 JSONL") from exc
            try:
                if deadline is None:
                    chunk = child.stdout.get()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _RpcIdleTimeout
                    chunk = child.stdout.get(timeout=remaining)
            except queue.Empty as exc:
                raise _RpcIdleTimeout from exc
            if chunk is None:
                raise _RpcProcessExited
            child.stdout_buffer.extend(chunk)

    def _raise_transport_or_host(
        self,
        operation: _RpcOperation,
        message: str,
        started: float,
        cause: BaseException | None,
        child: _RpcChild,
    ) -> None:
        if operation == "prompt":
            self._raise_ask_error(
                _transport_cause(cause),
                message,
                started,
                child,
                include_protocol_message=isinstance(cause, _RpcProtocolError),
            )
        raise SessionHostError(message, _operation_name(operation))

    def _raise_ask_error(
        self,
        cause: AgentTransportFailureCause,
        message: str,
        started: float,
        child: _RpcChild,
        *,
        include_protocol_message: bool = False,
    ) -> None:
        elapsed = time.monotonic() - started
        raise SessionAskError(
            cause=cause,
            exit_code=child.process.poll(),
            stderr_tail=stderr_tail(_stderr(child, message, include_protocol_message)),
            elapsed=elapsed,
            call_info=AgentCallInfo(
                argv=child.command.copy(),
                prompt_via_stdin=True,
                elapsed=elapsed,
                exit_code=child.process.poll(),
                sandboxed=child.prepared is not None,
                permission_mode=self._settings.permission_mode.value,
            ),
        )


class _RpcIdleTimeout(Exception):
    """No complete JSONL event arrived before the configured deadline."""


class _RpcProcessExited(Exception):
    """Pi closed its stdout before completing the active request."""


class _RpcProtocolError(Exception):
    """Pi emitted data that does not satisfy its JSONL protocol."""


def _start_reader(
    stream: IO[bytes],
    consume: Callable[[bytes], None],
    finish: Callable[[], None],
    stopped: threading.Event,
) -> threading.Thread:
    def read() -> None:
        try:
            while not stopped.is_set() and (chunk := os.read(stream.fileno(), 4096)):
                consume(chunk)
        finally:
            stream.close()
            finish()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    return reader


def _start_stderr_reader(child: _RpcChild, stream: IO[bytes]) -> threading.Thread:
    """Drain stderr while holding the delivery lock through each read and callback."""
    descriptor = stream.fileno()
    os.set_blocking(descriptor, False)

    def read() -> None:
        try:
            while not child.stopped.is_set():
                try:
                    readable = select.select([descriptor], [], [], 0.05)[0]
                except (OSError, ValueError):
                    break
                if not readable:
                    continue
                lock = child.stderr_lock
                lock.acquire()
                try:
                    if child.stopped.is_set():
                        break
                    try:
                        chunk = os.read(descriptor, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    _capture_stderr(child, chunk)
                finally:
                    lock.release()
        finally:
            lock = child.stderr_lock
            lock.acquire()
            try:
                stream.close()
                child.stderr_reader_done.set()
            finally:
                lock.release()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    return reader


def _queue_stdout(child: _RpcChild, chunk: bytes | None) -> None:
    while not child.stopped.is_set():
        try:
            child.stdout.put(chunk, timeout=0.1)
            return
        except queue.Full:
            pass


def _write_command(
    child: _RpcChild, command: dict[str, object], idle_timeout: float | None = None
) -> None:
    stdin = child.stdin
    data = (
        json.dumps(command, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    if idle_timeout is None:
        stdin.write(data)
        stdin.flush()
        return

    descriptor = stdin.fileno()
    deadline = time.monotonic() + idle_timeout
    view = memoryview(data)
    os.set_blocking(descriptor, False)
    try:
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [descriptor], [], remaining)[1]:
                raise _RpcIdleTimeout
            view = view[os.write(descriptor, view) :]
    finally:
        os.set_blocking(descriptor, True)


def _spawn(
    agent: AgentPi,
    command: list[str],
    operation: str,
    settings: BackendSettings,
) -> _RpcChild:
    env = settings.env
    sandbox_run = sandbox_run_for(settings.sandbox, settings.get_sandbox_context)
    prepared: PreparedSandboxCommand | None = None
    argv = command
    if sandbox_run is not None:
        spec = sandbox_run.limits.for_command(profile_name(command[0]) if command else None)
        try:
            prepared = sandbox_run.context.prepare(command, spec, env=env, cwd=Path.cwd())
        except (SandboxUnavailableError, SandboxSettingsError, FileNotFoundError) as exc:
            raise SessionHostError(f"could not start Pi RPC session: {exc}", operation) from exc
        argv = prepared.argv
    try:
        process: subprocess.Popen[bytes] = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
            start_new_session=True,
            env=prepared.env if prepared is not None else env,
            cwd=prepared.cwd if prepared is not None else None,
        )
    except (OSError, ValueError) as exc:
        if prepared is not None:
            prepared.close()
        raise SessionHostError(f"could not start Pi RPC session: {exc}", operation) from exc
    child = _RpcChild(process, agent, argv, prepared=prepared)
    child.readers.extend(
        (
            _start_reader(
                cast(IO[bytes], process.stdout),
                lambda chunk: _queue_stdout(child, chunk),
                lambda: _queue_stdout(child, None),
                child.stopped,
            ),
            _start_stderr_reader(child, cast(IO[bytes], process.stderr)),
        )
    )
    return child


def _terminate(child: _RpcChild) -> None:
    child.stopped.set()
    try:
        child.stdin.close()
    except OSError:
        pass
    prepared = child.prepared
    try:
        stop_process(
            child.process,
            isolate_process_group=True,
            interrupt_cleanup_cmd=prepared.interrupt_cleanup_cmd if prepared is not None else None,
            cwd=prepared.cwd if prepared is not None else None,
            env=prepared.env if prepared is not None else None,
        )
    finally:
        # ``stop_process`` can itself raise (e.g. a cleanup command whose
        # binary is missing); the sandbox's tracked artifacts and reader
        # threads must still be released rather than leaked.
        if prepared is not None:
            prepared.close()
        for reader in child.readers:
            reader.join(timeout=1)
        # The reader callbacks close over ``child``; dropping them breaks that
        # cycle so the process, queued stdout, and stderr tail are freed at once.
        child.readers.clear()


def _validate_response(event: dict[str, object]) -> None:
    identifier = event.get("id")
    command = event.get("command")
    success = event.get("success")
    if not isinstance(identifier, str) or not identifier:
        raise _RpcProtocolError("Pi RPC response had no id")
    if not isinstance(command, str) or not command:
        raise _RpcProtocolError("Pi RPC response had no command")
    if not isinstance(success, bool):
        raise _RpcProtocolError("Pi RPC response had no boolean success")
    if success is False:
        error = event.get("error")
        if not isinstance(error, str) or not error:
            raise _RpcProtocolError("Pi RPC failure response had no error")


def _event_assistant_text(event: dict[str, object]) -> str | None:
    if event["type"] != "message_end":
        return None
    raw_message = event.get("message")
    if not isinstance(raw_message, dict):
        return None
    message = cast(dict[str, object], raw_message)
    if message.get("role") != "assistant":
        return None
    content = message.get("content")
    if not isinstance(content, list):
        raise _RpcProtocolError("Pi RPC assistant message had malformed content")
    text: list[str] = []
    for raw_block in content:
        if not isinstance(raw_block, dict):
            raise _RpcProtocolError("Pi RPC assistant message had a malformed content block")
        block = cast(dict[str, object], raw_block)
        if block.get("type") == "text":
            value = block.get("text")
            if not isinstance(value, str):
                raise _RpcProtocolError("Pi RPC assistant message text block was not text")
            text.append(value)
    return "".join(text)


def _response_streaming_state(response: dict[str, object]) -> bool:
    data = response.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("isStreaming"), bool):
        raise _RpcProtocolError("Pi RPC returned malformed streaming state")
    return cast(bool, data["isStreaming"])


def _event_text_delta(event: dict[str, object]) -> str | None:
    if event["type"] != "message_update":
        return None
    update = event.get("assistantMessageEvent")
    if not isinstance(update, dict):
        raise _RpcProtocolError("Pi RPC message update had no assistant event")
    update_type = update.get("type")
    if not isinstance(update_type, str) or not update_type:
        raise _RpcProtocolError("Pi RPC message update had no assistant event type")
    if update_type != "text_delta":
        return None
    delta = update.get("delta")
    if not isinstance(delta, str):
        raise _RpcProtocolError("Pi RPC text delta was not text")
    return delta


def _capture_stderr(child: _RpcChild, chunk: bytes) -> None:
    """Retain Pi diagnostics and echo decoded stderr while an ask is active."""
    child.stderr.append(chunk)
    text = child.stderr_decoder.decode(chunk)
    callback = child.output_callback
    if callback is not None and text:
        try:
            callback("stderr", text)
        except OSError:
            pass


def _finish_stderr_delivery(child: _RpcChild, *, drain: bool = True) -> None:
    """Deliver pending diagnostics before disabling the active ask callback."""
    lock = child.stderr_lock
    lock.acquire()
    try:
        if drain and not child.stderr_reader_done.is_set():
            stream = child.process.stderr
            if stream is not None:
                try:
                    descriptor = stream.fileno()
                    while select.select([descriptor], [], [], 0)[0]:
                        chunk = os.read(descriptor, 4096)
                        if not chunk:
                            break
                        _capture_stderr(child, chunk)
                except (OSError, ValueError):
                    pass
        child.output_callback = None
    finally:
        lock.release()


def _emit_pi_progress(
    event: dict[str, object],
    *,
    delta: str | None,
    assistant_progress: list[str],
    tool_output_lengths: dict[str, str],
    tool_names: dict[str, str] | None = None,
    callback: AgentOutputCallback,
) -> None:
    """Echo Pi tool activity and assistant messages that invoke a tool."""
    event_type = event["type"]
    if event_type == "message_update" and delta is not None:
        assistant_progress.append(delta)
        return
    if event_type == "message_end":
        message = event.get("message")
        if isinstance(message, dict):
            typed_message = cast(dict[str, object], message)
        else:
            typed_message = None
        if typed_message is not None and typed_message.get("role") == "assistant":
            stop_reason = typed_message.get("stopReason")
            if stop_reason not in ("stop", "length", "error"):
                text = "".join(assistant_progress)
                if text:
                    callback("progress", text, event_type="message")
        assistant_progress.clear()
        return
    if event_type == "tool_execution_start":
        name = event.get("toolName")
        call_id = event.get("toolCallId")
        if tool_names is not None and isinstance(call_id, str) and isinstance(name, str):
            tool_names[call_id] = name
        if isinstance(name, str) and name:
            callback(
                "progress",
                f"[{name}]\n",
                event_type="tool_call",
                tool_name=name,
                tool_call_id=call_id if isinstance(call_id, str) else None,
            )
        return
    if event_type not in {"tool_execution_update", "tool_execution_end"}:
        return
    call_id = event.get("toolCallId")
    if not isinstance(call_id, str):
        return
    result = (
        event.get("partialResult") if event_type == "tool_execution_update" else event.get("result")
    )
    current = _rpc_result_text(result)
    previous = tool_output_lengths.get(call_id, "")
    added = current[len(previous) :] if current.startswith(previous) else current
    tool_output_lengths[call_id] = current
    if added:
        callback(
            "progress",
            added,
            event_type="tool_result",
            tool_name=None if tool_names is None else tool_names.get(call_id),
            tool_call_id=call_id,
        )


def _rpc_result_text(value: object) -> str:
    """Extract readable text blocks from one Pi tool result."""
    if not isinstance(value, dict):
        return ""
    content = value.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        cast(str, item_value["text"])
        for item in content
        if isinstance(item, dict)
        for item_value in (cast(dict[str, object], item),)
        if item_value.get("type") == "text" and isinstance(item_value.get("text"), str)
    )


def _extension_ui_cancellation(event: dict[str, object]) -> dict[str, object] | None:
    """Cancel extension dialog requests that AGM cannot present to a user."""
    if event["type"] != "extension_ui_request":
        return None
    method = event.get("method")
    if method not in {"confirm", "select", "input", "editor"}:
        return None
    request_id = event.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise _RpcProtocolError("Pi RPC extension UI request had no id")
    return {"type": "extension_ui_response", "id": request_id, "cancelled": True}


def _terminal_prompt_failure(event: dict[str, object]) -> str | None:
    event_type = event["type"]
    if event.get("willRetry") is True:
        return None
    if event_type == "auto_retry_end" and event.get("success") is False:
        return _event_error_message(event, "Pi RPC automatic retry failed")
    if event_type == "compaction_end" and event.get("willRetry") is not True:
        if event.get("result") is None and event.get("aborted") is False:
            return _event_error_message(event, "Pi RPC compaction failed")
    messages: object
    if event_type == "message_end":
        messages = [event.get("message")]
    elif event_type == "agent_end":
        messages = event.get("messages")
    else:
        return None
    if not isinstance(messages, list):
        raise _RpcProtocolError("Pi RPC terminal event had no messages")
    for message in messages:
        if not isinstance(message, dict):
            raise _RpcProtocolError("Pi RPC terminal event had a malformed message")
        stop_reason = message.get("stopReason")
        if stop_reason in {"error", "aborted"}:
            return _event_error_message(message, f"Pi RPC agent {stop_reason}")
    return None


def _event_error_message(event: dict[str, object], fallback: str) -> str:
    for key in ("errorMessage", "finalError", "error"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def _session_id(response: dict[str, object]) -> str | None:
    data = response.get("data")
    if not isinstance(data, dict):
        return None
    value = data.get("sessionId")
    if isinstance(value, str) and value:
        return value
    return None


def _required_session_id(response: dict[str, object], operation: str) -> str:
    session_id = _session_id(response)
    if session_id is None:
        raise _RpcProtocolError("Pi RPC did not report a session id")
    return session_id


def _replacement_session_id(response: dict[str, object], operation: str, parent_id: str) -> str:
    session_id = _required_session_id(response, operation)
    if session_id != parent_id:
        raise _RpcProtocolError("Pi RPC replacement session id did not match its parent")
    return session_id


def _forked_session_id(response: dict[str, object], operation: str, parent_id: str) -> str:
    session_id = _required_session_id(response, operation)
    if session_id == parent_id:
        raise _RpcProtocolError("Pi RPC clone did not create an independent session")
    return session_id


def _require_not_cancelled(response: dict[str, object], operation: str) -> None:
    data = response.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("cancelled"), bool):
        raise _RpcProtocolError("Pi RPC returned malformed cancellation state")
    if data["cancelled"] is True:
        raise SessionHostError("Pi RPC cancelled the session operation", operation)


def _stats_from_response(response: dict[str, object]) -> SessionStats:
    data = response.get("data")
    if not isinstance(data, dict):
        raise _malformed_stats()
    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        raise _malformed_stats()
    input_tokens = tokens.get("input")
    output_tokens = tokens.get("output")
    if (
        type(input_tokens) is not int
        or type(output_tokens) is not int
        or input_tokens < 0
        or output_tokens < 0
    ):
        raise _malformed_stats()
    cost = _bounded_decimal(data.get("cost"))
    if cost is None or cost < 0:
        raise _malformed_stats()
    context_usage = data.get("contextUsage")
    context_percent: Decimal
    if context_usage is None:
        context_percent = _CONTEXT_PERCENT_UNAVAILABLE
    elif isinstance(context_usage, dict):
        percent = context_usage.get("percent")
        if percent is None:
            context_percent = _CONTEXT_PERCENT_UNAVAILABLE
        else:
            parsed_percent = _bounded_decimal(percent)
            if parsed_percent is None or parsed_percent < 0 or parsed_percent > 100:
                raise _malformed_stats()
            context_percent = parsed_percent
    else:
        raise _malformed_stats()
    return SessionStats(input_tokens, output_tokens, cost, context_percent)


def _bounded_decimal(value: object) -> Decimal | None:
    """Parse *value* into a decimal within the pinned AgL range, or ``None``.

    Malformed, non-finite, and out-of-range values are all treated alike by
    the caller: as malformed session stats, since a runtime ``DecimalValue``
    built from them must stay within the same range as any other.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, str)):
        return None
    try:
        decimal = parse_json_decimal(str(value))
    except ValueError:
        return None
    return decimal if decimal_in_range(decimal) else None


def _malformed_stats() -> _RpcProtocolError:
    return _RpcProtocolError("Pi RPC returned malformed session statistics")


def _operation_name(operation: _RpcOperation) -> str:
    return {
        "prompt": SessionOperation.ASK.value,
        "compact": SessionOperation.COMPACT.value,
        "new_session": "reset",
        "clone": SessionOperation.FORK.value,
        "set_session_name": SessionOperation.SET_NAME.value,
        "get_session_stats": SessionOperation.STATS.value,
        "get_state": SessionOperation.FORK.value,
    }[operation]


def _transport_cause(error: BaseException | None) -> AgentTransportFailureCause:
    if isinstance(error, _RpcIdleTimeout):
        return "timeout"
    if isinstance(error, _RpcProtocolError):
        return "protocol_failure"
    return "nonzero_exit"


def _stderr(child: _RpcChild, fallback: str, include_fallback: bool = False) -> str:
    text, note = child.stderr.captured().text_or_note("stderr")
    if not text:
        return f"{fallback}; {note}" if note else fallback
    if include_fallback:
        return f"{text}\n{fallback}"
    return text
