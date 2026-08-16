"""Unit tests for CLI-backed agent session adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agent.runner import command_targets_session_id
from agm.agent.session import (
    SessionAskError,
    SessionAskRequest,
    SessionHostError,
    SessionOpenRequest,
    SessionService,
)
from agm.agent.session.cli_adapters import AgentCommandSessionBackend
from agm.agent.spec import AgentCommand
from agm.core.process import ProcessCaptureResult
from agm.util.interp import InterpolationError


def _open(backend: AgentCommandSessionBackend, command: str, *, name: str = "") -> None:
    backend.open(SessionOpenRequest(agent=AgentCommand(command), transport="cli", name=name))


def _capture_result() -> ProcessCaptureResult:
    return ProcessCaptureResult(
        returncode=0,
        stdout="answer",
        stderr="",
        elapsed=0.1,
        timed_out=False,
        spawn_error=None,
        spawn_errno=None,
    )


def _non_prompt_args(argv: list[str]) -> list[str]:
    return [arg for arg in argv if not arg.startswith("@")]


def test_open_requires_an_agent_command() -> None:
    backend = AgentCommandSessionBackend()

    with pytest.raises(SessionHostError) as raised:
        backend.open(SessionOpenRequest(agent=object(), transport="cli"))

    assert raised.value.operation == "open"


def test_open_requires_a_well_formed_command() -> None:
    backend = AgentCommandSessionBackend()

    with pytest.raises(SessionHostError) as raised:
        _open(backend, 'runner "unterminated %{SESSION_ID}')

    assert raised.value.operation == "open"


def test_open_requires_a_session_id_placeholder() -> None:
    backend = AgentCommandSessionBackend()

    with pytest.raises(SessionHostError) as raised:
        _open(backend, "runner --quiet")

    assert raised.value.operation == "open"


def test_one_shot_command_session_does_not_require_a_session_id_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run_capture_result(argv: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured.append(argv)
        return _capture_result()

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)
    service = SessionService(lambda _agent, _transport: AgentCommandSessionBackend())

    response = service.ask_ephemeral(
        AgentCommand("runner --quiet"), "cli", SessionAskRequest(prompt="question")
    )

    assert response.content == "answer"
    assert _non_prompt_args(captured[0]) == ["runner", "--quiet"]


def test_open_converts_malformed_placeholder_to_an_open_error() -> None:
    backend = AgentCommandSessionBackend()

    with pytest.raises(SessionHostError) as raised:
        _open(backend, "runner --session=%{SESSION_ID} --invalid=%{")

    assert raised.value.operation == "open"


def test_command_session_target_check_adds_the_malformed_element_context() -> None:
    with pytest.raises(InterpolationError) as raised:
        command_targets_session_id(["runner", "--invalid=%{"])

    assert raised.value.context == "in command element '--invalid=%{'"


def test_open_does_not_accept_an_escaped_session_id_placeholder() -> None:
    backend = AgentCommandSessionBackend()

    with pytest.raises(SessionHostError) as raised:
        _open(backend, r"runner --session='\%{SESSION_ID}'")

    assert raised.value.operation == "open"


def test_ask_preserves_interpolation_failures_as_ask_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGM_SESSION_TEST_MISSING", raising=False)
    backend = AgentCommandSessionBackend()
    _open(backend, "runner --session=%{SESSION_ID} --option=%{AGM_SESSION_TEST_MISSING}")

    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest(prompt="question"))

    assert raised.value.cause == "interpolation_failure"
    assert raised.value.call_info.argv == [
        "runner",
        "--session=%{SESSION_ID}",
        "--option=%{AGM_SESSION_TEST_MISSING}",
    ]


def test_open_rejects_a_backend_visible_name() -> None:
    backend = AgentCommandSessionBackend()

    with pytest.raises(SessionHostError) as raised:
        _open(backend, "runner --session %{SESSION_ID}", name="named")

    assert raised.value.operation == "set-name"


def test_asks_reuse_one_underlying_id_with_a_symmetric_command_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run_capture_result(argv: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured.append(argv)
        return _capture_result()

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)
    backend = AgentCommandSessionBackend()
    _open(backend, "runner --session %{SESSION_ID}")

    first = backend.ask(SessionAskRequest(prompt="first"))
    second = backend.ask(SessionAskRequest(prompt="second"))

    assert first.content == "answer"
    assert second.content == "answer"
    assert len(captured) == 2
    assert _non_prompt_args(captured[0]) == _non_prompt_args(captured[1])
    assert captured[0][1] == captured[1][1]


def test_ask_interpolates_mixed_prompt_session_and_escaped_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run_capture_result(argv: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured.append(argv)
        return _capture_result()

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)
    backend = AgentCommandSessionBackend()
    _open(
        backend,
        r"runner --session=%{SESSION_ID} --literal='\%{SESSION_ID}' "
        "--input=%{PROMPT_FILE}",
    )

    backend.ask(SessionAskRequest(prompt="question"))

    assert captured[0][1].startswith("--session=")
    assert captured[0][2] == "--literal=%{SESSION_ID}"
    assert captured[0][3].startswith("--input=")
    assert not any(arg.startswith("@") for arg in captured[0])


@pytest.mark.parametrize(
    ("cause", "returncode", "timed_out", "spawn_error"),
    [
        ("spawn_failure", None, False, "not found"),
        ("timeout", None, True, None),
        ("nonzero_exit", 2, False, None),
    ],
)
def test_ask_preserves_failed_process_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    cause: str,
    returncode: int | None,
    timed_out: bool,
    spawn_error: str | None,
) -> None:
    stderr = "" if spawn_error is not None else "runner failure " * 50

    def fake_run_capture_result(argv: list[str], **kwargs: object) -> ProcessCaptureResult:
        return ProcessCaptureResult(
            returncode=returncode,
            stdout="partial answer",
            stderr=stderr,
            elapsed=0.1,
            timed_out=timed_out,
            spawn_error=spawn_error,
            spawn_errno=None,
        )

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)
    backend = AgentCommandSessionBackend()
    _open(backend, "runner --session %{SESSION_ID}")

    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest(prompt="question"))

    error = raised.value
    assert error.cause == cause
    assert error.exit_code == returncode
    assert error.stderr_tail == (stderr or spawn_error or "")[-500:]
    assert error.elapsed == 0.1
    assert error.call_info.argv[:2] == ["runner", "--session"]
    assert error.call_info.prompt_via_stdin is False
    assert error.call_info.elapsed == 0.1
    assert error.call_info.exit_code == returncode


def test_ask_preserves_failed_process_diagnostics_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_capture_result(argv: list[str], **kwargs: object) -> ProcessCaptureResult:
        return ProcessCaptureResult(
            returncode=2,
            stdout="partial answer",
            stderr="runner failure",
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
            spawn_errno=None,
        )

    def failing_cleanup(temp_files: list[Path]) -> None:
        raise OSError("cleanup failed")

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)
    monkeypatch.setattr("agm.agent.session.cli_adapters.cleanup_temp_files", failing_cleanup)
    backend = AgentCommandSessionBackend()
    _open(backend, "runner --session %{SESSION_ID}")

    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest(prompt="question"))

    assert raised.value.cause == "nonzero_exit"
    assert raised.value.exit_code == 2
    assert raised.value.stderr_tail == "runner failure"


def test_ask_preserves_an_interpolation_failure_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[list[Path]] = []

    def failing_cleanup(temp_files: list[Path]) -> None:
        cleanup_calls.append(temp_files)
        raise OSError("cleanup failed")

    monkeypatch.delenv("AGM_SESSION_TEST_MISSING", raising=False)
    monkeypatch.setattr("agm.agent.session.cli_adapters.cleanup_temp_files", failing_cleanup)
    backend = AgentCommandSessionBackend()
    _open(backend, "runner --session=%{SESSION_ID} --option=%{AGM_SESSION_TEST_MISSING}")

    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest(prompt="question"))

    assert raised.value.cause == "interpolation_failure"
    assert cleanup_calls


def test_ask_cleans_up_after_an_interrupt_without_masking_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[list[Path]] = []

    def interrupted_run_capture_result(argv: list[str], **kwargs: object) -> ProcessCaptureResult:
        raise KeyboardInterrupt

    def failing_cleanup(temp_files: list[Path]) -> None:
        cleanup_calls.append(temp_files)
        raise OSError("cleanup failed")

    monkeypatch.setattr("agm.agent.runner.run_capture_result", interrupted_run_capture_result)
    monkeypatch.setattr("agm.agent.session.cli_adapters.cleanup_temp_files", failing_cleanup)
    backend = AgentCommandSessionBackend()
    _open(backend, "runner --session %{SESSION_ID}")

    with pytest.raises(KeyboardInterrupt):
        backend.ask(SessionAskRequest(prompt="question"))

    assert cleanup_calls


def test_ask_surfaces_cleanup_failure_after_a_successful_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_cleanup(temp_files: list[Path]) -> None:
        raise OSError("cleanup failed")

    def fake_run_capture_result(argv: list[str], **kwargs: object) -> ProcessCaptureResult:
        return _capture_result()

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)
    monkeypatch.setattr("agm.agent.session.cli_adapters.cleanup_temp_files", failing_cleanup)
    backend = AgentCommandSessionBackend()
    _open(backend, "runner --session %{SESSION_ID}")

    with pytest.raises(OSError, match="cleanup failed"):
        backend.ask(SessionAskRequest(prompt="question"))


def test_reset_mints_a_new_underlying_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run_capture_result(argv: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured.append(argv)
        return _capture_result()

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)
    backend = AgentCommandSessionBackend()
    _open(backend, "runner --session %{SESSION_ID}")

    backend.ask(SessionAskRequest(prompt="first"))
    backend.reset()
    backend.ask(SessionAskRequest(prompt="second"))

    assert captured[0][2] != captured[1][2]


@pytest.mark.parametrize(
    ("operation", "invoke"),
    [
        ("compact", lambda service, handle: service.compact(handle)),
        ("fork", lambda service, handle: service.fork(handle)),
        ("stats", lambda service, handle: service.stats(handle)),
        ("set-name", lambda service, handle: service.set_name(handle, "renamed")),
    ],
)
def test_unsupported_operations_are_rejected_by_the_session_service(
    operation: str,
    invoke: object,
) -> None:
    backend = AgentCommandSessionBackend()
    service = SessionService(lambda agent, transport: backend)
    handle = service.open(AgentCommand("runner --session %{SESSION_ID}"), "cli")

    if not callable(invoke):
        raise AssertionError("test operation must be callable")
    with pytest.raises(SessionHostError) as raised:
        invoke(service, handle)

    assert raised.value.operation == operation


@pytest.mark.parametrize(
    ("method", "args", "operation"),
    [
        ("compact", ("make room",), "compact"),
        ("fork", (), "fork"),
        ("set_name", ("renamed",), "set-name"),
        ("stats", (), "stats"),
    ],
)
def test_direct_unsupported_operations_raise_capability_errors(
    method: str,
    args: tuple[str, ...],
    operation: str,
) -> None:
    backend = AgentCommandSessionBackend()
    _open(backend, "runner --session %{SESSION_ID}")

    with pytest.raises(SessionHostError) as raised:
        getattr(backend, method)(*args)

    assert raised.value.operation == operation


def test_close_drops_the_underlying_command_state() -> None:
    backend = AgentCommandSessionBackend()
    _open(backend, "runner --session %{SESSION_ID}")

    backend.close()

    with pytest.raises(SessionHostError) as raised:
        backend.ask(SessionAskRequest(prompt="again"))

    assert raised.value.operation == "ask"
