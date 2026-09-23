"""Session-host cleanup at the ``exec`` command boundary."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from agm.agent.session import (
    AglSessionHost,
    SessionAskRequest,
    SessionAskResponse,
    SessionCapabilities,
    SessionHostError,
    SessionOpenRequest,
    SessionService,
    SessionStats,
)
from agm.agent.spec import PermissionMode
from agm.commands import exec_program
from agm.sandbox.request import SandboxLimits
from tests._agl_helpers import write_file_program
from tests.test_exec_command import file_args


class _SessionHost:
    def __init__(self, *, close_error: Exception | None = None) -> None:
        self.close_calls = 0
        self.close_error = close_error

    def open(
        self,
        _agent: object,
        _transport: str,
        *,
        name: str = "",
        permission_mode: PermissionMode = PermissionMode.NONE,
        sandbox: SandboxLimits | None = None,
    ) -> str:
        del name, permission_mode, sandbox
        return "session"

    def close_all(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def test_exec_injects_and_closes_the_session_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "session.agl"
    write_file_program(path, 'let session = Session::open(AgentPi("p", "m", "high"))')
    host = _SessionHost()
    monkeypatch.setattr(exec_program, "create_agl_session_host", lambda **_kwargs: host)

    assert exec_program.run(file_args(path)) is None
    # The interpreter owns live sessions; the command's outer cleanup makes
    # pre-execution and unexpected failures safe as well.
    assert host.close_calls == 2


def test_exec_keeps_a_failed_run_result_primary_when_session_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "failed-session.agl"
    path.write_text(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentPi("p", "m", "high"))\n'
        '  raise RangeError(message = "primary")\n',
        encoding="utf-8",
    )
    host = _SessionHost(close_error=RuntimeError("cleanup failed"))
    monkeypatch.setattr(exec_program, "create_agl_session_host", lambda **_kwargs: host)

    with pytest.raises(SystemExit) as exited:
        exec_program.run(file_args(path))

    assert exited.value.code == 2
    assert any("cleanup failed" in note for note in exited.value.__notes__)


class _InterruptingBackend:
    capabilities = SessionCapabilities.all()

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.reset_calls = 0

    def open(self, request: SessionOpenRequest) -> None:
        del request
        self.opened = True

    def ask(self, request: SessionAskRequest) -> SessionAskResponse:
        del request
        raise AssertionError("the program must interrupt before asking")

    def compact(self, instructions: str) -> None:
        del instructions
        raise AssertionError("the program must interrupt before compacting")

    def reset(self) -> None:
        self.reset_calls += 1
        os.kill(os.getpid(), signal.SIGINT)
        raise AssertionError("SIGINT did not interrupt interpreter execution")

    def fork(self) -> _InterruptingBackend:
        raise AssertionError("the program must interrupt before forking")

    def set_name(self, name: str) -> None:
        del name
        raise AssertionError("the program must interrupt before setting a name")

    def stats(self) -> SessionStats:
        raise AssertionError("the program must interrupt before reading stats")

    def close(self) -> None:
        self.closed = True


class _RecordingSessionService(SessionService):
    def __init__(self, backend: _InterruptingBackend) -> None:
        super().__init__(lambda _agent, _transport: backend)
        self.opened_handle: str | None = None

    def open(
        self,
        agent: object,
        transport: str,
        *,
        name: str = "",
        ephemeral: bool = False,
        single_prompt: bool = False,
        permission_mode: PermissionMode = PermissionMode.NONE,
        sandbox: SandboxLimits | None = None,
    ) -> str:
        handle = super().open(
            agent,
            transport,
            name=name,
            ephemeral=ephemeral,
            single_prompt=single_prompt,
            permission_mode=permission_mode,
            sandbox=sandbox,
        )
        self.opened_handle = handle
        return handle


def test_exec_sigint_closes_an_opened_session_and_restores_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, default_sigint: None
) -> None:
    path = tmp_path / "session.agl"
    write_file_program(
        path,
        'let session = Session::open(AgentCommand("worker"))\nsession.reset()',
    )
    backend = _InterruptingBackend()
    service = _RecordingSessionService(backend)
    host = AglSessionHost(service)
    monkeypatch.setattr(exec_program, "create_agl_session_host", lambda **_kwargs: host)
    previous_handler = signal.getsignal(signal.SIGINT)

    with pytest.raises(KeyboardInterrupt):
        exec_program.run(file_args(path))

    assert backend.opened
    assert backend.reset_calls == 1
    assert backend.closed
    assert service.opened_handle is not None
    with pytest.raises(SessionHostError):
        service.reset(service.opened_handle)
    assert signal.getsignal(signal.SIGINT) is previous_handler
