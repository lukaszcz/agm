"""CLI-backed implementations of agent session backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generic, Self, TypeVar, assert_never, cast
from uuid import uuid4

from agm.agent import runner
from agm.agent.runner import (
    PromptDelivery,
    cleanup_temp_files,
    command_targets_session_id,
    prompt_run_result_error,
)
from agm.agent.session.protocol import (
    SessionAgentError,
    SessionAskError,
    SessionAskRequest,
    SessionAskResponse,
    SessionBackend,
    SessionHostError,
    SessionOpenRequest,
    SessionOperation,
    SessionOperations,
)
from agm.agent.spec import AgentClaude, AgentCodex, AgentCommand, AgentPi
from agm.agent.transport import AgentCallInfo, AgentTransportFailureCause, stderr_tail
from agm.core.cleanup import preserve_primary_error
from agm.core.env import clone_env
from agm.util.interp import InterpolationError
from agm.util.unicode import loads_json


@dataclass(slots=True)
class _CommandSession:
    """The command and underlying id held by an open command session."""

    command: list[str]
    session_id: str


_SessionAgentT = TypeVar("_SessionAgentT", AgentClaude, AgentPi)


@dataclass(slots=True)
class _SessionIdCliState(Generic[_SessionAgentT]):
    """The common local lifecycle state for a CLI session with a generated id."""

    agent: _SessionAgentT
    session_id: str
    name: str
    single_prompt: bool
    started: bool = False


@dataclass(slots=True)
class _CodexSession:
    """The state needed to start or resume a Codex CLI transcript."""

    agent: AgentCodex
    single_prompt: bool
    session_id: str | None = None
    started: bool = False


def _creation_not_launched(error: SessionAskError) -> bool:
    """Whether an initial ask failed before a transcript process could start."""
    return error.cause in {"spawn_failure", "interpolation_failure"}


def open_cli_session(request: SessionOpenRequest, *, idle_timeout: float | None) -> SessionBackend:
    """Open the CLI session backend for *request*'s agent variant."""
    name = request.name
    single_prompt = request.single_prompt
    match request.agent:
        case AgentCommand() as command:
            return AgentCommandSessionBackend.open(
                command, name=name, single_prompt=single_prompt, idle_timeout=idle_timeout
            )
        case AgentClaude() as claude:
            return ClaudeCliSessionBackend.open(
                claude, name=name, single_prompt=single_prompt, idle_timeout=idle_timeout
            )
        case AgentCodex() as codex:
            return CodexCliSessionBackend.open(
                codex, name=name, single_prompt=single_prompt, idle_timeout=idle_timeout
            )
        case AgentPi() as pi:
            return PiCliSessionBackend.open(
                pi, name=name, single_prompt=single_prompt, idle_timeout=idle_timeout
            )
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


class _CliPromptBackend:
    """Shared prepared-runner boundary for CLI session implementations."""

    def __init__(self, *, idle_timeout: float | None) -> None:
        self._idle_timeout = idle_timeout

    @property
    def operations(self) -> SessionOperations:
        """No optional operations; backends that implement some override this."""
        return SessionOperations()

    def close(self) -> None:
        """Nothing to release: a CLI session holds only local state between prompts."""

    def _run_prompt(
        self,
        prompt: str,
        command: list[str],
        *,
        delivery: PromptDelivery = PromptDelivery.FILE,
        session_id: str | None = None,
    ) -> SessionAskResponse:
        """Run one prepared prompt and translate process failures for sessions."""
        temp_files: list[Path] = []
        with preserve_primary_error(
            lambda: cleanup_temp_files(temp_files), label="Temporary prompt cleanup"
        ):
            try:
                prepared = runner.prepare_rendered_prompt_run(
                    prompt,
                    runner=command,
                    temp_files=temp_files,
                    env=clone_env(),
                    delivery=delivery,
                    session_id=session_id,
                )
                result = runner.run_prepared_prompt_result(
                    prepared, idle_timeout=self._idle_timeout
                )
            except InterpolationError as exc:
                raise SessionAskError(
                    cause="interpolation_failure",
                    exit_code=None,
                    stderr_tail=str(exc),
                    elapsed=0.0,
                    call_info=AgentCallInfo(
                        argv=command.copy(),
                        prompt_via_stdin=delivery is PromptDelivery.STDIN,
                        elapsed=0.0,
                        exit_code=None,
                    ),
                ) from exc
            if failure := prompt_run_result_error(result):
                raise SessionAskError(
                    cause=failure.cause,
                    exit_code=failure.result.returncode,
                    stderr_tail=stderr_tail(
                        failure.result.stderr.text_or_note("stderr")[0]
                        or failure.result.spawn_error
                        or ""
                    ),
                    elapsed=failure.result.elapsed,
                    call_info=AgentCallInfo(
                        argv=(prepared.argv or []).copy(),
                        prompt_via_stdin=prepared.prompt_via_stdin,
                        elapsed=failure.result.elapsed,
                        exit_code=failure.result.returncode,
                    ),
                    detail=failure.detail,
                ) from failure
            return SessionAskResponse(
                content=result.stdout.text(),
                metadata={"elapsed": result.elapsed},
                call_info=AgentCallInfo(
                    argv=(prepared.argv or []).copy(),
                    prompt_via_stdin=prepared.prompt_via_stdin,
                    elapsed=result.elapsed,
                    exit_code=result.returncode,
                ),
            )


class AgentCommandSessionBackend(_CliPromptBackend):
    """Run an ``AgentCommand`` repeatedly with its generated session id."""

    def __init__(self, session: _CommandSession, *, idle_timeout: float | None) -> None:
        super().__init__(idle_timeout=idle_timeout)
        self._session = session

    @classmethod
    def open(
        cls,
        agent: AgentCommand,
        *,
        name: str = "",
        single_prompt: bool = False,
        idle_timeout: float | None = None,
    ) -> Self:
        """Validate *agent*'s command and open a session over it."""
        if name:
            raise SessionHostError("command sessions do not support names", "open")
        try:
            command = agent.argv()
        except ValueError as exc:
            raise SessionAgentError(str(exc), "open") from exc
        try:
            targets_session_id = command_targets_session_id(command)
        except InterpolationError as exc:
            raise SessionAgentError(str(exc), "open") from exc
        if not targets_session_id and not single_prompt:
            raise SessionHostError(
                "command session requires a %{SESSION_ID} placeholder; "
                "use a single-attempt AgentCommand.ask instead",
                "open",
            )
        return cls(
            _CommandSession(command=command, session_id=str(uuid4())), idle_timeout=idle_timeout
        )

    def ask(self, request: SessionAskRequest) -> SessionAskResponse:
        """Run the command with this session's id and the rendered prompt."""
        return self._run_prompt(
            request.prompt,
            self._session.command,
            delivery=PromptDelivery.FILE,
            session_id=self._session.session_id,
        )

    def reset(self) -> None:
        """Replace the underlying id while retaining this backend instance."""
        self._session.session_id = str(uuid4())


class _SessionIdCliBackend(_CliPromptBackend, Generic[_SessionAgentT], ABC):
    """Common lifecycle for CLI backends that generate and retain a session id."""

    def __init__(
        self, session: _SessionIdCliState[_SessionAgentT], *, idle_timeout: float | None
    ) -> None:
        super().__init__(idle_timeout=idle_timeout)
        self._session: _SessionIdCliState[_SessionAgentT] = session

    @classmethod
    def open(
        cls,
        agent: _SessionAgentT,
        *,
        name: str = "",
        single_prompt: bool = False,
        idle_timeout: float | None = None,
    ) -> Self:
        """Allocate the id used by the backend's first prompt."""
        return cls(
            _SessionIdCliState(agent, str(uuid4()), name, single_prompt),
            idle_timeout=idle_timeout,
        )

    def ask(self, request: SessionAskRequest) -> SessionAskResponse:
        """Start or resume this backend's transcript for one prompt."""
        session = self._session
        command = self._prompt_command(session)
        was_started = session.started
        session.started = True
        try:
            return self._run_prompt(request.prompt, command, delivery=PromptDelivery.FILE)
        except SessionAskError as error:
            if not was_started and _creation_not_launched(error):
                session.started = False
            raise

    def reset(self) -> None:
        """Discard the transcript while retaining this backend instance."""
        self._session.session_id = str(uuid4())
        self._session.started = False

    @abstractmethod
    def _prompt_command(self, session: _SessionIdCliState[_SessionAgentT]) -> list[str]:
        """Build the backend-specific command for the current session state."""

    def _forked(self, session_id: str) -> Self:
        """Return a backend for an already-created native fork of this transcript."""
        return type(self)(
            _SessionIdCliState(self._session.agent, session_id, "", False, started=True),
            idle_timeout=self._idle_timeout,
        )

    def _unstarted_fork(self) -> Self:
        """Fork deferred local state before either transcript exists natively."""
        session = self._session
        return type(self)(
            _SessionIdCliState(
                session.agent, str(uuid4()), "", session.single_prompt, started=False
            ),
            idle_timeout=self._idle_timeout,
        )


class ClaudeCliSessionBackend(_SessionIdCliBackend[AgentClaude]):
    """Continue Claude conversations through its CLI session flags."""

    @property
    def operations(self) -> SessionOperations:
        """Claude compacts and forks natively."""
        return SessionOperations(compact=self.compact, fork=self.fork)

    def _prompt_command(self, session: _SessionIdCliState[AgentClaude]) -> list[str]:
        return (
            session.agent.argv()
            if session.single_prompt
            else session.agent.session_argv(
                session.session_id,
                resume=session.started,
                name=session.name if not session.started else "",
            )
        )

    def compact(self, instructions: str) -> None:
        """Ask Claude to compact the current transcript and confirm the result."""
        session = self._session
        if not session.started:
            return
        prompt = "/compact" if not instructions else f"/compact {instructions}"
        response = self._run_prompt(
            prompt,
            session.agent.session_argv(session.session_id, resume=True, json_output=True),
            delivery=PromptDelivery.LITERAL,
        )
        _require_claude_compaction_confirmation(response.content)
        session.started = True

    def fork(self) -> ClaudeCliSessionBackend:
        """Fork the current Claude transcript and return its child backend."""
        session = self._session
        if not session.started:
            return self._unstarted_fork()
        response = self._run_prompt(
            "",
            session.agent.session_argv(
                session.session_id, resume=True, fork=True, json_output=True
            ),
            delivery=PromptDelivery.NONE,
        )
        return self._forked(_require_claude_session_id(response.content))


class CodexCliSessionBackend(_CliPromptBackend):
    """Start Codex threads from JSONL, then resume them through plaintext output."""

    def __init__(self, session: _CodexSession, *, idle_timeout: float | None) -> None:
        super().__init__(idle_timeout=idle_timeout)
        self._session = session

    @classmethod
    def open(
        cls,
        agent: AgentCodex,
        *,
        name: str = "",
        single_prompt: bool = False,
        idle_timeout: float | None = None,
    ) -> Self:
        """Allocate a deferred Codex session without starting a thread."""
        if name:
            raise SessionHostError("Codex CLI sessions do not support names", "open")
        return cls(_CodexSession(agent, single_prompt), idle_timeout=idle_timeout)

    def ask(self, request: SessionAskRequest) -> SessionAskResponse:
        """Start a Codex thread once, then resume its captured id.

        A failed turn is a recoverable agent failure: the ask still fails, but
        the session stays usable, resuming the announced thread when the stream
        named one and otherwise starting a fresh thread on the next ask.
        """
        session = self._session
        if session.single_prompt:
            return self._run_prompt(
                request.prompt, session.agent.argv(), delivery=PromptDelivery.STDIN
            )
        if session.session_id is None and session.started:
            raise SessionHostError(
                "Codex session start did not produce a resumable thread", SessionOperation.ASK.value
            )
        starting = session.session_id is None
        session.started = True
        try:
            response = self._run_prompt(
                request.prompt,
                session.agent.session_argv(session.session_id),
                delivery=PromptDelivery.STDIN,
            )
        except SessionAskError as error:
            if starting and _creation_not_launched(error):
                session.started = False
            raise
        if not starting:
            return response
        try:
            thread_id, content = _parse_codex_jsonl(response.content)
        except _CodexTurnFailedError as exc:
            if exc.thread_id is None:
                session.started = False
            else:
                session.session_id = exc.thread_id
            raise _codex_ask_error(response, "nonzero_exit", exc) from exc
        except _CodexProtocolError as exc:
            raise _codex_ask_error(response, "protocol_failure", exc) from exc
        session.session_id = thread_id
        return replace(response, content=content)

    def reset(self) -> None:
        """Forget the captured thread so the next prompt starts a new one."""
        self._session.session_id = None
        self._session.started = False


class PiCliSessionBackend(_SessionIdCliBackend[AgentPi]):
    """Continue Pi conversations through its CLI session flags."""

    @property
    def operations(self) -> SessionOperations:
        """Pi's CLI forks natively."""
        return SessionOperations(fork=self.fork)

    def _prompt_command(self, session: _SessionIdCliState[AgentPi]) -> list[str]:
        return (
            session.agent.argv()
            if session.single_prompt
            else session.agent.session_argv(
                session.session_id,
                name=session.name if not session.started else "",
            )
        )

    def fork(self) -> PiCliSessionBackend:
        """Snapshot this transcript natively and return the live child backend."""
        session = self._session
        if not session.started:
            return self._unstarted_fork()
        child_id = str(uuid4())
        self._run_prompt(
            "",
            session.agent.session_argv(child_id, fork_from=session.session_id),
            delivery=PromptDelivery.NONE,
        )
        return self._forked(child_id)


def _require_claude_compaction_confirmation(output: str) -> None:
    """Require Claude's JSON result to confirm a successful compaction."""
    payload = _json_object(output, operation=SessionOperation.COMPACT)
    if payload.get("is_error") is not False:
        raise SessionHostError(
            "Claude did not confirm session compaction", SessionOperation.COMPACT.value
        )


def _require_claude_session_id(output: str) -> str:
    """Extract the new session id from Claude's JSON fork result."""
    payload = _json_object(output, operation=SessionOperation.FORK)
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise SessionHostError(
            "Claude did not report a fork session id", SessionOperation.FORK.value
        )
    return session_id


def _parse_codex_jsonl(output: str) -> tuple[str, str]:
    """Validate Codex's JSONL event stream and extract its completed reply.

    The stream belongs to Codex, not to AGM, so unrecognized event types are
    ignored rather than rejected; only a failed turn and a malformed stream
    stop the parse.
    """
    thread_id: str | None = None
    messages: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            event = loads_json(line)
        except json.JSONDecodeError as exc:
            raise _CodexProtocolError("Codex returned malformed JSONL") from exc
        if not isinstance(event, dict):
            raise _CodexProtocolError("Codex JSONL event was not an object")
        event_object = cast(dict[str, object], event)
        event_type = event_object.get("type")
        if not isinstance(event_type, str):
            raise _CodexProtocolError("Codex JSONL event had no type")
        if event_type == "thread.started":
            candidate = event_object.get("thread_id")
            if thread_id is not None or not isinstance(candidate, str) or not candidate:
                raise _CodexProtocolError("Codex reported an invalid thread id")
            thread_id = candidate
        elif event_type == "turn.failed":
            raise _CodexTurnFailedError(_codex_turn_failure(event_object), thread_id)
        elif event_type == "item.completed":
            item = event_object.get("item")
            if not isinstance(item, dict):
                raise _CodexProtocolError("Codex completed item was malformed")
            item_object = cast(dict[str, object], item)
            item_type = item_object.get("type")
            if not isinstance(item_type, str):
                raise _CodexProtocolError("Codex completed item was malformed")
            if item_type == "agent_message":
                text = item_object.get("text")
                if not isinstance(text, str):
                    raise _CodexProtocolError("Codex assistant message had no text")
                messages.append(text)
    if thread_id is None:
        raise _CodexProtocolError("Codex did not report a session id")
    if not messages:
        raise _CodexProtocolError("Codex did not complete an assistant message")
    return thread_id, "\n".join(messages)


def _codex_turn_failure(event: dict[str, object]) -> str:
    """Return the reason Codex reported for a failed turn."""
    error = event.get("error")
    if isinstance(error, dict):
        message = cast(dict[str, object], error).get("message")
        if isinstance(message, str) and message:
            return message
    return "Codex reported a failed turn without a reason"


def _codex_ask_error(
    response: SessionAskResponse, cause: AgentTransportFailureCause, exc: Exception
) -> SessionAskError:
    """Build the ask failure for a started Codex thread that produced no reply."""
    call_info = cast(AgentCallInfo, response.call_info)
    return SessionAskError(
        cause=cause,
        exit_code=call_info.exit_code,
        stderr_tail=stderr_tail(str(exc)),
        elapsed=call_info.elapsed,
        call_info=call_info,
    )


class _CodexProtocolError(Exception):
    """Codex output did not satisfy its JSONL response protocol."""


class _CodexTurnFailedError(Exception):
    """Codex reported a failed turn through its JSONL event stream.

    The thread id is the one the stream announced before the failure, if any;
    it lets a recoverable turn failure leave the thread resumable.
    """

    def __init__(self, reason: str, thread_id: str | None) -> None:
        super().__init__(reason)
        self.thread_id = thread_id


def _json_object(output: str, *, operation: SessionOperation) -> dict[str, object]:
    """Decode the one JSON object returned by a Claude lifecycle command."""
    try:
        payload: object = loads_json(output)
    except json.JSONDecodeError as exc:
        raise SessionHostError(
            f"Claude did not return JSON for {operation.value}", operation.value
        ) from exc
    if not isinstance(payload, dict):
        raise SessionHostError(
            f"Claude did not return a JSON object for {operation.value}", operation.value
        )
    return cast(dict[str, object], payload)
