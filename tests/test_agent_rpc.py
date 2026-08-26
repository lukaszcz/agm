"""Deterministic protocol tests for the persistent Pi RPC session backend."""

from __future__ import annotations

import json
import os
import sys
import threading
from decimal import Decimal
from pathlib import Path
from time import monotonic, sleep
from typing import cast

import pytest

from agm.agent.session import (
    SessionAskError,
    SessionAskRequest,
    SessionHostError,
    SessionOpenRequest,
    SessionService,
    SessionStats,
    rpc,
)
from agm.agent.session.rpc import PiRpcSessionBackend
from agm.agent.spec import AgentPi

_STUB = r"""#!{python}
import json, os, sys, time
from pathlib import Path
root = Path(os.environ["PI_RPC_STUB_ROOT"])
settings = json.loads((root / "script.json").read_text())
actions = settings.get("actions", {})
argv = sys.argv[1:]
session = argv[argv.index("--session-id") + 1] if "--session-id" in argv else "root"
def record(name, value):
    with (root / name).open("a") as out:
        out.write(json.dumps(value) + "\n")
        out.flush()
record("starts.jsonl", {"pid": os.getpid(), "argv": argv, "session": session})
def expand(value, command):
    if isinstance(value, str):
        return value.replace("$id", command["id"]).replace(
            "$command", command["type"]
        )
    if isinstance(value, list): return [expand(item, command) for item in value]
    if isinstance(value, dict): return {key: expand(item, command) for key, item in value.items()}
    return value
def response(command):
    result = {"id": command["id"], "type": "response", "command": command["type"], "success": True}
    if command["type"] == "get_state":
        result["data"] = {"sessionId": session, "isStreaming": False}
    if command["type"] == "new_session": result["data"] = {"cancelled": False}
    if command["type"] == "clone": result["data"] = {"cancelled": False}
    if command["type"] == "get_session_stats":
        result["data"] = settings.get("stats") or {
            "tokens": {"input": 11, "output": 7},
            "cost": 0.125,
            "contextUsage": {"percent": 3.5},
        }
    return result
def emit(event):
    if "raw" in event:
        sys.stdout.buffer.write(event["raw"].encode() + b"\n")
    else:
        sys.stdout.buffer.write(json.dumps(event, allow_nan=True).encode() + b"\n")
    sys.stdout.buffer.flush()
for line in sys.stdin.buffer:
    command = json.loads(line)
    record("commands.jsonl", command)
    action = actions.get(command["type"])
    if action == "wait":
        (root / ("waiting-" + command["type"])).touch()
        while not (root / ("release-" + command["type"])).exists(): time.sleep(.01)
    elif action == "exit":
        sys.exit(0)
    events = action if isinstance(action, list) else None
    if events is None:
        if command["type"] == "prompt":
            events = [
                response(command),
                {"type": "message_update", "assistantMessageEvent": {
                    "type": "text_delta", "delta": "answer",
                }},
                {"type": "agent_settled"},
            ]
        else:
            events = [response(command)]
    for event in expand(events, command):
        if event == {"exit": True}: sys.exit(0)
        emit(event)
        if (
            command["type"] == "clone"
            and event.get("type") == "response"
            and event.get("success") is True
            and event.get("data", {}).get("cancelled") is False
        ):
            session = "child"
"""


class RpcStub:
    """A small stateful JSONL Pi child controlled through files in one tmp directory."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        actions: dict[str, object] | None = None,
        *,
        stats: dict[str, object] | None = None,
    ) -> None:
        self.root = tmp_path / "rpc"
        self.root.mkdir()
        script_actions = dict[str, object]() if actions is None else actions
        script: dict[str, object] = {"actions": script_actions, "stats": stats}
        (self.root / "script.json").write_text(json.dumps(script))
        pi = self.root / "pi"
        pi.write_text(_STUB.replace("{python}", sys.executable))
        pi.chmod(0o755)
        monkeypatch.setenv("PI_RPC_STUB_ROOT", str(self.root))
        monkeypatch.setenv("PATH", f"{self.root}{os.pathsep}{os.environ['PATH']}")

    def records(self, name: str) -> list[dict[str, object]]:
        path = self.root / name
        return (
            []
            if not path.exists()
            else [
                cast(dict[str, object], json.loads(line)) for line in path.read_text().splitlines()
            ]
        )

    def wait_for(self, name: str, count: int = 1) -> list[dict[str, object]]:
        deadline = monotonic() + 5
        while monotonic() < deadline:
            records = self.records(name)
            if len(records) >= count:
                return records
            sleep(0.01)
        raise AssertionError(f"stub did not write {name}")

    def wait_for_file(self, name: str) -> None:
        deadline = monotonic() + 5
        while monotonic() < deadline:
            if (self.root / name).exists():
                return
            sleep(0.01)
        raise AssertionError(f"stub did not write {name}")

    def release(self, command: str) -> None:
        (self.root / f"release-{command}").touch()


def open_backend(*, timeout: float | None = None) -> PiRpcSessionBackend:
    backend = PiRpcSessionBackend(idle_timeout=timeout)
    backend.open(SessionOpenRequest(AgentPi("provider", "model", "high"), "rpc", "named"))
    return backend


def option_value(argv: object, option: str) -> str | None:
    """Return the value *option* carries in *argv*, or ``None`` when absent."""
    assert isinstance(argv, list)
    if option not in argv:
        return None
    return str(argv[argv.index(option) + 1])


def command_types(stub: RpcStub) -> list[str]:
    return [str(record["type"]) for record in stub.records("commands.jsonl")]


def assert_exited(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_prompt_handled_without_agent_run_completes_and_keeps_session_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = RpcStub(
        tmp_path,
        monkeypatch,
        {
            "prompt": [{"id": "$id", "type": "response", "command": "prompt", "success": True}],
            "get_state": [
                {
                    "id": "$id",
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {"sessionId": "root", "isStreaming": False},
                }
            ],
        },
    )
    backend = open_backend(timeout=0.2)

    assert backend.ask(SessionAskRequest("handled command")).content == ""
    backend.compact("")

    assert command_types(stub) == ["prompt", "get_state", "compact"]
    backend.close()


def test_interrupting_prompt_kills_the_active_rpc_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = RpcStub(tmp_path, monkeypatch)
    backend = open_backend()
    pid = cast(int, stub.wait_for("starts.jsonl")[0]["pid"])

    monkeypatch.setattr(
        backend, "_next_event", lambda _child: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    with pytest.raises(KeyboardInterrupt):
        backend.ask(SessionAskRequest("interrupt"))

    assert_exited(pid)
    with pytest.raises(SessionHostError):
        backend.compact("")
    backend.close()


@pytest.mark.parametrize(
    ("command_type", "error"),
    [
        ("prompt", KeyboardInterrupt()),
        ("get_state", KeyboardInterrupt()),
        ("get_state", rpc._RpcIdleTimeout()),
        ("get_state", OSError("closed")),
    ],
)
def test_prompt_write_interruption_or_failure_kills_rpc_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command_type: str,
    error: BaseException,
) -> None:
    stub = RpcStub(
        tmp_path,
        monkeypatch,
        {"prompt": [{"id": "$id", "type": "response", "command": "prompt", "success": True}]},
    )
    backend = open_backend()
    pid = cast(int, stub.wait_for("starts.jsonl")[0]["pid"])
    write_command = rpc._write_command

    def fail_selected_write(
        child: rpc._RpcChild, command: dict[str, object], idle_timeout: float | None
    ) -> None:
        if command["type"] == command_type:
            raise error
        write_command(child, command, idle_timeout)

    monkeypatch.setattr(rpc, "_write_command", fail_selected_write)
    expected = KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else SessionAskError

    with pytest.raises(expected):
        backend.ask(SessionAskRequest("interrupt"))

    assert_exited(pid)
    backend.close()


def test_spawn_prompt_and_lifecycle_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = RpcStub(
        tmp_path,
        monkeypatch,
        {
            "prompt": [
                {"type": "agent_start"},
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": "first "},
                },
                {"id": "$id", "type": "response", "command": "$command", "success": True},
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": "answer"},
                },
                {"type": "agent_settled"},
            ]
        },
    )
    backend = open_backend()
    argv = stub.wait_for("starts.jsonl")[0]["argv"]
    assert argv == [
        "--mode",
        "rpc",
        "--name",
        "named",
        "--provider",
        "provider",
        "--model",
        "model",
        "--thinking",
        "high",
    ]
    assert backend.ask(SessionAskRequest("hello")).content == "first answer"
    backend.compact("")
    backend.compact("retain")
    backend.reset()
    backend.set_name("new")
    assert backend.stats() == SessionStats(11, 7, Decimal("0.125"), Decimal("3.5"))
    commands = stub.records("commands.jsonl")
    assert "customInstructions" not in commands[2]
    assert commands[3]["customInstructions"] == "retain"
    assert command_types(stub) == [
        "prompt",
        "get_state",
        "compact",
        "compact",
        "new_session",
        "set_session_name",
        "get_session_stats",
    ]
    backend.close()


@pytest.mark.parametrize("method", ["confirm", "select", "input", "editor"])
def test_prompt_cancels_extension_ui_dialog_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    stub = RpcStub(
        tmp_path,
        monkeypatch,
        {
            "prompt": [
                {"id": "$id", "type": "response", "command": "prompt", "success": True},
                {"type": "extension_ui_request", "id": "ui-1", "method": method},
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": "answer"},
                },
                {"type": "agent_settled"},
            ]
        },
    )
    backend = open_backend()

    assert backend.ask(SessionAskRequest("hello")).content == "answer"

    assert stub.wait_for("commands.jsonl", 3)[2] == {
        "id": "ui-1",
        "type": "extension_ui_response",
        "cancelled": True,
    }
    backend.close()


def test_extension_ui_cancellation_write_failure_is_an_ask_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(
        tmp_path,
        monkeypatch,
        {
            "prompt": [
                {"id": "$id", "type": "response", "command": "prompt", "success": True},
                {"type": "extension_ui_request", "id": "ui-1", "method": "confirm"},
            ]
        },
    )
    backend = open_backend()
    write_command = rpc._write_command

    def fail_ui_cancellation(
        child: rpc._RpcChild, command: dict[str, object], idle_timeout: float | None
    ) -> None:
        if command["type"] == "extension_ui_response":
            raise BrokenPipeError
        write_command(child, command, idle_timeout)

    monkeypatch.setattr(rpc, "_write_command", fail_ui_cancellation)

    with pytest.raises(SessionAskError):
        backend.ask(SessionAskRequest("hello"))

    backend.close()


def test_clone_immediately_after_open_keeps_parent_and_child_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = RpcStub(tmp_path, monkeypatch)
    backend = open_backend()

    child = backend.fork()

    starts = stub.wait_for("starts.jsonl", 2)
    parent_argv, replacement_argv = (record["argv"] for record in starts)
    assert parent_argv == [
        "--mode",
        "rpc",
        "--name",
        "named",
        "--provider",
        "provider",
        "--model",
        "model",
        "--thinking",
        "high",
    ]
    assert option_value(replacement_argv, "--session-id") == "root"
    assert command_types(stub) == ["get_state", "get_state", "clone", "get_state"]
    assert child.ask(SessionAskRequest("child")).content == "answer"
    assert backend.ask(SessionAskRequest("parent")).content == "answer"
    backend.close()
    child.close()


def test_clone_does_not_reapply_the_startup_name_to_the_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = RpcStub(tmp_path, monkeypatch)
    backend = open_backend()
    backend.set_name("renamed")

    child = backend.fork()

    replacement_argv = stub.wait_for("starts.jsonl", 2)[1]["argv"]
    assert "--name" not in replacement_argv
    assert option_value(replacement_argv, "--session-id") == "root"
    backend.close()
    child.close()


def test_clone_snapshots_current_branch_and_keeps_both_children_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = RpcStub(tmp_path, monkeypatch)
    backend = open_backend()
    backend.ask(SessionAskRequest("before"))
    child = backend.fork()
    starts = stub.wait_for("starts.jsonl", 2)
    assert option_value(starts[1]["argv"], "--session-id") == "root"
    assert command_types(stub) == [
        "prompt",
        "get_state",
        "get_state",
        "get_state",
        "clone",
        "get_state",
    ]
    assert child.ask(SessionAskRequest("child")).content == "answer"
    assert backend.ask(SessionAskRequest("parent")).content == "answer"
    backend.close()
    child.close()


@pytest.mark.parametrize("command", ["new_session", "clone"])
def test_cancelled_session_operations_are_rejected_and_do_not_transfer_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    stub = RpcStub(
        tmp_path,
        monkeypatch,
        {
            command: [
                {
                    "id": "$id",
                    "type": "response",
                    "command": "$command",
                    "success": True,
                    "data": {"cancelled": True},
                }
            ]
        },
    )
    backend = open_backend()
    with pytest.raises(SessionHostError):
        backend.reset() if command == "new_session" else backend.fork()
    assert backend.ask(SessionAskRequest("still parent")).content == "answer"
    if command == "clone":
        assert command_types(stub) == ["get_state", "get_state", "clone", "prompt", "get_state"]
    backend.close()


@pytest.mark.parametrize(
    ("action", "error"),
    [
        (
            [
                {
                    "id": "$id",
                    "type": "response",
                    "command": "prompt",
                    "success": False,
                    "error": "bad",
                }
            ],
            SessionAskError,
        ),
        (
            [
                {
                    "id": "$id",
                    "type": "response",
                    "command": "get_session_stats",
                    "success": False,
                    "error": "bad",
                }
            ],
            SessionHostError,
        ),
    ],
)
def test_error_ack_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: list[object], error: type[Exception]
) -> None:
    command = "prompt" if error is SessionAskError else "get_session_stats"
    RpcStub(tmp_path, monkeypatch, {command: action})
    backend = open_backend()
    with pytest.raises(error):
        backend.ask(SessionAskRequest("hello")) if command == "prompt" else backend.stats()
    backend.close()


def test_idle_timeout_waits_for_explicit_child_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = RpcStub(tmp_path, monkeypatch, {"prompt": "wait"})
    backend = open_backend(timeout=0.5)
    done = threading.Event()
    errors: list[Exception] = []

    def ask() -> None:
        try:
            backend.ask(SessionAskRequest("hello"))
        except Exception as error:
            errors.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=ask)
    thread.start()
    stub.wait_for_file("waiting-prompt")
    assert done.wait(2)
    thread.join()
    assert isinstance(errors[0], SessionAskError)
    assert errors[0].cause == "timeout"
    with pytest.raises(SessionHostError):
        backend.stats()
    backend.close()


def test_idle_timeout_covers_blocked_rpc_stdin_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pi = tmp_path / "pi"
    pi.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    pi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    backend = open_backend(timeout=0.05)

    started = monotonic()
    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest("x" * 1_000_000))

    assert raised.value.cause == "timeout"
    assert monotonic() - started < 2
    with pytest.raises(SessionHostError):
        backend.stats()
    backend.close()


def test_process_death_retains_stderr_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(tmp_path, monkeypatch, {"prompt": "exit"})
    backend = open_backend()
    child = backend._child
    assert child is not None
    child.stderr.append("useful diagnostic")
    with pytest.raises(SessionAskError) as raised:
        backend.ask(SessionAskRequest("hello"))
    assert raised.value.stderr_tail == "useful diagnostic"
    with pytest.raises(SessionHostError):
        backend.ask(SessionAskRequest("again"))
    backend.close()


def test_process_death_between_asks_makes_later_operations_lifecycle_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(tmp_path, monkeypatch)
    backend = open_backend()
    assert backend.ask(SessionAskRequest("first")).content == "answer"
    child = backend._child
    assert child is not None
    child.process.terminate()
    child.process.wait(timeout=5)

    with pytest.raises(SessionHostError) as compact_error:
        backend.compact("")
    assert compact_error.value.operation == "compact"
    with pytest.raises(SessionHostError) as ask_error:
        backend.ask(SessionAskRequest("second"))
    assert ask_error.value.operation == "ask"
    backend.close()


def test_close_and_close_all_terminate_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = RpcStub(tmp_path, monkeypatch)
    backend = open_backend()
    pid = stub.wait_for("starts.jsonl")[0]["pid"]
    assert isinstance(pid, int)
    backend.close()
    backend.close()
    assert_exited(pid)
    service = SessionService(lambda _agent, _transport: PiRpcSessionBackend())
    handle = service.open(AgentPi("", "", ""), "rpc")
    child_pid = stub.wait_for("starts.jsonl", 2)[1]["pid"]
    service.close_all()
    service.close_all()
    assert isinstance(child_pid, int)
    assert_exited(child_pid)
    with pytest.raises(SessionHostError):
        service.ask(handle, SessionAskRequest("no"))


def test_close_terminates_rpc_process_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descendant_file = tmp_path / "descendant"
    pi = tmp_path / "pi"
    pi.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys, time\n"
        f"path = {str(descendant_file)!r}\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "open(path, 'w').write(str(child.pid))\n"
        "while True: time.sleep(1)\n"
    )
    pi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    backend = open_backend()
    deadline = monotonic() + 5
    while not descendant_file.exists() and monotonic() < deadline:
        sleep(0.01)
    descendant_pid = int(descendant_file.read_text())

    backend.close()

    deadline = monotonic() + 5
    while monotonic() < deadline:
        try:
            os.kill(descendant_pid, 0)
        except ProcessLookupError:
            break
        status = Path(f"/proc/{descendant_pid}/stat")
        if status.exists() and status.read_text().split()[2] == "Z":
            break
        sleep(0.01)
    else:
        os.kill(descendant_pid, 9)
        raise AssertionError("RPC descendant survived backend close")


def test_stats_accepts_unavailable_context_and_rejects_invalid_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    RpcStub(
        tmp_path,
        monkeypatch,
        stats={"tokens": {"input": 0, "output": 0}, "cost": 0, "contextUsage": None},
    )
    backend = open_backend()
    assert backend.stats().context_percent == Decimal(0)
    backend.close()
    for value in (float("nan"), float("inf"), -1, 101):
        with pytest.raises(rpc._RpcProtocolError):
            rpc._stats_from_response(
                {
                    "data": {
                        "tokens": {"input": 1, "output": 2},
                        "cost": 1,
                        "contextUsage": {"percent": value},
                    }
                }
            )
