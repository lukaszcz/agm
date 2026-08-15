"""CLI-backed implementations of agent session backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from agm.agent.runner import (
    cleanup_temp_files,
    command_targets_session_id,
    prepare_rendered_prompt_run,
    prompt_run_result_error,
    run_prepared_prompt_result,
)
from agm.agent.session.protocol import (
    SessionAskError,
    SessionAskRequest,
    SessionAskResponse,
    SessionCapabilities,
    SessionHostError,
    SessionOpenRequest,
    SessionOperation,
    SessionStats,
)
from agm.agent.spec import AgentCommand
from agm.agent.transport import AgentCallInfo, stderr_tail
from agm.core.env import clone_env
from agm.util.interp import InterpolationError


@dataclass(slots=True)
class _CommandSession:
    """The command and underlying id held by an open command session."""

    command: list[str]
    session_id: str


def _cleanup_after_ask(temp_files: list[Path], primary_error: BaseException | None = None) -> None:
    """Clean up a prompt, retaining any error already raised by the ask."""
    try:
        cleanup_temp_files(temp_files)
    except BaseException as cleanup_error:
        if primary_error is None:
            raise
        primary_error.add_note(f"Temporary prompt cleanup also failed: {cleanup_error}")


class AgentCommandSessionBackend:
    """Run an ``AgentCommand`` repeatedly with its generated session id."""

    capabilities = SessionCapabilities(frozenset({SessionOperation.ASK}))

    def __init__(self, *, idle_timeout: float | None = None) -> None:
        self._idle_timeout = idle_timeout
        self._session: _CommandSession | None = None

    def open(self, request: SessionOpenRequest) -> None:
        """Validate and initialize a command session."""
        if request.name:
            raise SessionHostError(
                "command sessions do not support names",
                SessionOperation.SET_NAME.value,
            )
        if not isinstance(request.agent, AgentCommand):
            raise SessionHostError("command session requires an AgentCommand", "open")
        try:
            command = request.agent.argv()
        except ValueError as exc:
            raise SessionHostError(str(exc), "open") from exc
        try:
            targets_session_id = command_targets_session_id(command)
        except InterpolationError as exc:
            raise SessionHostError(str(exc), "open") from exc
        if not targets_session_id:
            raise SessionHostError(
                "command session requires a %{SESSION_ID} placeholder",
                "open",
            )
        self._session = _CommandSession(command=command, session_id=str(uuid4()))

    def ask(self, request: SessionAskRequest) -> SessionAskResponse:
        """Run the command with this session's id and the rendered prompt."""
        session = self._session_for("ask")
        temp_files: list[Path] = []
        try:
            try:
                prepared = prepare_rendered_prompt_run(
                    request.prompt,
                    runner=session.command,
                    temp_files=temp_files,
                    env=clone_env(),
                    session_id=session.session_id,
                )
                result = run_prepared_prompt_result(prepared, idle_timeout=self._idle_timeout)
            except InterpolationError as exc:
                raise SessionAskError(
                    cause="interpolation_failure",
                    exit_code=None,
                    stderr_tail=str(exc),
                    elapsed=0.0,
                    call_info=AgentCallInfo(
                        argv=session.command.copy(),
                        prompt_via_stdin=False,
                        elapsed=0.0,
                        exit_code=None,
                    ),
                ) from exc
            if failure := prompt_run_result_error(result):
                raise SessionAskError(
                    cause=failure.cause,
                    exit_code=failure.result.returncode,
                    stderr_tail=stderr_tail(
                        failure.result.stderr or failure.result.spawn_error or ""
                    ),
                    elapsed=failure.result.elapsed,
                    call_info=AgentCallInfo(
                        argv=(prepared.argv or []).copy(),
                        prompt_via_stdin=prepared.prompt_via_stdin,
                        elapsed=failure.result.elapsed,
                        exit_code=failure.result.returncode,
                    ),
                ) from failure
        except BaseException as primary_error:
            _cleanup_after_ask(temp_files, primary_error)
            raise
        _cleanup_after_ask(temp_files)
        return SessionAskResponse(content=result.stdout)

    def compact(self, instructions: str) -> None:
        """Reject unsupported command-session compaction."""
        raise SessionHostError("command sessions do not support compaction", "compact")

    def reset(self) -> None:
        """Replace the underlying id while retaining this backend instance."""
        session = self._session_for("reset")
        session.session_id = str(uuid4())

    def fork(self) -> AgentCommandSessionBackend:
        """Reject unsupported command-session forking."""
        raise SessionHostError("command sessions do not support forking", "fork")

    def set_name(self, name: str) -> None:
        """Reject unsupported command-session naming."""
        raise SessionHostError("command sessions do not support names", "set-name")

    def stats(self) -> SessionStats:
        """Reject unsupported command-session usage reporting."""
        raise SessionHostError("command sessions do not support statistics", "stats")

    def close(self) -> None:
        """Drop the command and underlying id."""
        self._session = None

    def _session_for(self, operation: str) -> _CommandSession:
        session = self._session
        if session is None:
            raise SessionHostError("command session is not open", operation)
        return session
