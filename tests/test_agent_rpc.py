"""Deterministic protocol tests for the persistent Pi RPC session backend."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from decimal import Decimal
from pathlib import Path
from time import monotonic, sleep
from typing import cast

import pytest

from agm.agent.session import (
    AglSessionHost,
    SessionAskError,
    SessionAskRequest,
    SessionHostError,
    SessionOpenRequest,
    SessionService,
    SessionStats,
    rpc,
)
from agm.agent.session.rpc import PiRpcSessionBackend
from agm.agent.spec import AgentPi, PermissionMode
from agm.sandbox.request import PreparedSandboxCommand, SandboxLimits
from tests._agl_helpers import (
    session_sandbox_context,
    unavailable_sandbox_context,
    write_sandbox_home,
)

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


def _install_transparent_sandbox_shims(directory: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install silent, flag-skipping ``systemd-run``/``srt`` fakes on the front of ``PATH``.

    Each touches a marker file under the returned directory before exec'ing
    onward, so a test can confirm the wrap chain actually ran, not merely
    that the wrapped ``pi`` stub happened to start anyway.
    """
    directory.mkdir(parents=True, exist_ok=True)
    log_dir = directory / "log"
    log_dir.mkdir()
    systemd_run = directory / "systemd-run"
    systemd_run.write_text(
        "#!/bin/bash\n"
        f'touch "{log_dir}/systemd-run"\n'
        "while [[ $# -gt 0 ]]; do\n"
        '  case "$1" in\n'
        "    --user|--scope|-q) shift ;;\n"
        "    -p|--unit) shift 2 ;;\n"
        '    --) shift; exec "$@" ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
    )
    systemd_run.chmod(0o755)
    srt = directory / "srt"
    srt.write_text(
        "#!/bin/bash\n"
        f'touch "{log_dir}/srt"\n'
        "while [[ $# -gt 0 ]]; do\n"
        '  case "$1" in\n'
        "    --settings) shift 2 ;;\n"
        '    --) shift; exec "$@" ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
    )
    srt.chmod(0o755)
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")
    return log_dir


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
    backend = PiRpcSessionBackend(
        idle_timeout=timeout, get_sandbox_context=unavailable_sandbox_context
    )
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


def test_prompt_reply_combines_a_surrogate_escape_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A high+low escape pair in the JSONL event text is one scalar character.

    json.dumps escapes an astral character as an adjacent ``\\uD83D\\uDE00``
    pair (ensure_ascii, the stub's default); ``_next_event`` must accept the
    combined result rather than treat it as a lone surrogate.
    """
    RpcStub(
        tmp_path,
        monkeypatch,
        {
            "prompt": [
                {"id": "$id", "type": "response", "command": "prompt", "success": True},
                {
                    "type": "message_update",
                    "assistantMessageEvent": {"type": "text_delta", "delta": "\U0001f600"},
                },
                {"type": "agent_settled"},
            ]
        },
    )
    backend = open_backend()

    assert backend.ask(SessionAskRequest("hello")).content == "\U0001f600"
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
    child.stderr.append(b"useful diagnostic")
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
    service = SessionService(
        lambda _agent, _transport: PiRpcSessionBackend(
            get_sandbox_context=unavailable_sandbox_context
        )
    )
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
        "with open(path + '.tmp', 'w') as file: file.write(str(child.pid))\n"
        "os.replace(path + '.tmp', path)\n"
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


def test_open_spawns_the_rpc_child_wrapped_under_sandbox_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persistent RPC child is prepared once, at spawn time, under the
    session's fixed sandbox mode -- the whole systemd-run/srt chain reaches
    the real ``pi`` process before any prompt is ever sent."""
    log_dir = _install_transparent_sandbox_shims(tmp_path / "shims", monkeypatch)
    home = tmp_path / "home"
    write_sandbox_home(home)
    stub = RpcStub(tmp_path, monkeypatch)

    backend = PiRpcSessionBackend(get_sandbox_context=session_sandbox_context(home))
    backend.open(
        SessionOpenRequest(
            AgentPi("provider", "model", "high"),
            "rpc",
            "named",
            permission_mode=PermissionMode.UNRESTRICTED,
            sandbox=SandboxLimits(),
        )
    )

    stub.wait_for("starts.jsonl")
    assert (log_dir / "systemd-run").exists()
    assert (log_dir / "srt").exists()
    backend.close()


def test_ephemeral_host_ask_spawns_the_rpc_child_sandboxed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ephemeral, one-shot ask (the shape a free/explicit-agent ``ask`` opens)
    reaches the real sandboxed spawn through the full ``AglSessionHost`` ->
    ``SessionService`` -> backend chain -- not only when a test opens the
    backend directly -- proving the mode a one-shot session opens under
    actually governs its Pi RPC child."""
    log_dir = _install_transparent_sandbox_shims(tmp_path / "shims", monkeypatch)
    home = tmp_path / "home"
    write_sandbox_home(home)
    RpcStub(tmp_path, monkeypatch)

    host = AglSessionHost(
        SessionService(
            lambda agent, transport: PiRpcSessionBackend(
                get_sandbox_context=session_sandbox_context(home)
            )
        )
    )

    answer = host.with_ephemeral(
        AgentPi("provider", "model", "high"),
        "rpc",
        lambda handle: host.ask(handle, "hi"),
        single_prompt=True,
        permission_mode=PermissionMode.UNRESTRICTED,
        sandbox=SandboxLimits(),
    )

    assert answer == "answer"
    assert (log_dir / "systemd-run").exists()
    assert (log_dir / "srt").exists()


def test_open_prepare_failure_becomes_a_session_host_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sandbox preparation failure at open -- no settings file resolvable --
    is a session lifecycle failure, never a bare spawn of an unsandboxed child."""
    RpcStub(tmp_path, monkeypatch)
    home = tmp_path / "home"  # No `.agm/sandbox/default.json` written.

    backend = PiRpcSessionBackend(get_sandbox_context=session_sandbox_context(home))

    with pytest.raises(SessionHostError) as raised:
        backend.open(
            SessionOpenRequest(
                AgentPi("provider", "model", "high"),
                "rpc",
                permission_mode=PermissionMode.UNRESTRICTED,
                sandbox=SandboxLimits(),
            )
        )
    assert raised.value.operation == "open"


def test_close_and_fork_replacement_close_the_prepared_sandbox_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``close()`` releases the prepared command backing the child it stops, and
    forking -- which spawns a freshly prepared replacement for the parent --
    leaves its own prepared command open only until that replacement itself closes."""
    _install_transparent_sandbox_shims(tmp_path / "shims", monkeypatch)
    home = tmp_path / "home"
    write_sandbox_home(home)
    # No scripted "get_state"/"clone" actions: the stub's own default responses
    # track its live ``session`` variable, which a fork must actually advance.
    RpcStub(tmp_path, monkeypatch)

    closed: list[bool] = []
    original_close = PreparedSandboxCommand.close

    def spy_close(self: PreparedSandboxCommand) -> None:
        closed.append(self._closed)
        original_close(self)

    monkeypatch.setattr(PreparedSandboxCommand, "close", spy_close)

    backend = PiRpcSessionBackend(get_sandbox_context=session_sandbox_context(home))
    backend.open(
        SessionOpenRequest(
            AgentPi("provider", "model", "high"),
            "rpc",
            permission_mode=PermissionMode.UNRESTRICTED,
            sandbox=SandboxLimits(),
        )
    )
    assert closed == []  # Nothing closed yet: the child is still alive.

    child = backend.fork()
    # Forking spawned a freshly prepared replacement for the still-active
    # parent conversation; the original prepared command transferred to
    # ``child`` without being re-prepared or closed.
    assert closed == []

    backend.close()
    assert closed == [False]  # The replacement's own prepared command closed.
    child.close()
    assert closed == [False, False]  # The transferred prepared command closed.


def test_spawn_closes_the_prepared_command_when_popen_itself_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spawn failure that happens after a successful sandbox preparation --
    ``Popen`` itself raising -- still releases the prepared command, exactly
    like a preparation failure does."""
    _install_transparent_sandbox_shims(tmp_path / "shims", monkeypatch)
    home = tmp_path / "home"
    write_sandbox_home(home)

    closed: list[bool] = []
    original_close = PreparedSandboxCommand.close

    def spy_close(self: PreparedSandboxCommand) -> None:
        closed.append(self._closed)
        original_close(self)

    monkeypatch.setattr(PreparedSandboxCommand, "close", spy_close)

    def fail_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        raise OSError("no such file")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)

    backend = PiRpcSessionBackend(get_sandbox_context=session_sandbox_context(home))

    with pytest.raises(SessionHostError) as raised:
        backend.open(
            SessionOpenRequest(
                AgentPi("provider", "model", "high"),
                "rpc",
                permission_mode=PermissionMode.UNRESTRICTED,
                sandbox=SandboxLimits(),
            )
        )
    assert raised.value.operation == "open"
    assert closed == [False]


def test_close_still_releases_the_prepared_command_when_stop_process_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``stop_process`` failure (e.g. a missing cleanup binary) must not leak
    the prepared sandbox command's temp settings file or tracked artifacts."""
    _install_transparent_sandbox_shims(tmp_path / "shims", monkeypatch)
    home = tmp_path / "home"
    write_sandbox_home(home)
    RpcStub(tmp_path, monkeypatch)

    closed: list[bool] = []
    original_close = PreparedSandboxCommand.close

    def spy_close(self: PreparedSandboxCommand) -> None:
        closed.append(self._closed)
        original_close(self)

    monkeypatch.setattr(PreparedSandboxCommand, "close", spy_close)

    def raising_stop_process(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("systemctl")

    monkeypatch.setattr(rpc, "stop_process", raising_stop_process)

    backend = PiRpcSessionBackend(get_sandbox_context=session_sandbox_context(home))
    backend.open(
        SessionOpenRequest(
            AgentPi("provider", "model", "high"),
            "rpc",
            permission_mode=PermissionMode.UNRESTRICTED,
            sandbox=SandboxLimits(),
        )
    )

    with pytest.raises(FileNotFoundError):
        backend.close()
    assert closed == [False]
