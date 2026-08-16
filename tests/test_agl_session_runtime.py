"""Runtime behavior for host-backed AgL sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import pytest

from agm.agl import PipelineDriver
from agm.agl.pipeline import RunResult
from agm.agl.runtime.request import (
    AgentCallHostError,
    AgentCallInfo,
    AgentCancelled,
    AgentRequest,
    AgentResponse,
)
from agm.agl.runtime.sessions import (
    AgentDispatcherSessionHost,
    SessionAskError,
    SessionHostError,
    SessionSnapshot,
    SessionStats,
    with_ephemeral_session,
)
from agm.agl.semantics.values import EnumValue
from tests._agl_helpers import agent_value


@dataclass
class _Host:
    handles: dict[str, tuple[EnumValue, str]] = field(default_factory=dict)
    prompts: dict[str, list[str]] = field(default_factory=dict)
    operations: list[tuple[str, str, str]] = field(default_factory=list)
    closed: set[str] = field(default_factory=set)
    default_handle: str | None = None

    def open(self, agent: EnumValue, transport: str, *, name: str = "") -> str:
        handle = f"s{len(self.handles) + 1}"
        self.handles[handle] = (agent, transport)
        self.prompts[handle] = []
        self.operations.append((handle, "open", name))
        return handle

    def open_ephemeral(self, agent: EnumValue, transport: str, *, one_shot: bool = False) -> str:
        del one_shot
        return self.open(agent, transport)

    def default(self, agent: EnumValue, transport: str, *, name: str = "") -> str:
        if self.default_handle is None:
            self.default_handle = self.open(agent, transport, name=name)
        return self.default_handle

    def ask(self, handle: str, prompt: str) -> str:
        self._live(handle, "ask")
        self.prompts[handle].append(prompt)
        return "answer"

    def compact(self, handle: str, instructions: str = "") -> None:
        self._live(handle, "compact")
        self.operations.append((handle, "compact", instructions))

    def reset(self, handle: str) -> None:
        self._live(handle, "reset")
        self.operations.append((handle, "reset", ""))

    def fork(self, handle: str) -> str:
        agent, transport = self._live(handle, "fork")
        child = self.open(agent, transport)
        self.operations.append((handle, "fork", child))
        return child

    def set_name(self, handle: str, name: str) -> None:
        self._live(handle, "set-name")
        self.operations.append((handle, "set-name", name))

    def stats(self, handle: str) -> SessionStats:
        self._live(handle, "stats")
        return SessionStats(3, 5, Decimal("0.12"), Decimal("4.5"))

    def snapshot(self, handle: str) -> SessionSnapshot:
        agent, transport = self.handles[handle]
        return SessionSnapshot(agent, transport)

    def close(self, handle: str) -> None:
        if handle not in self.handles:
            raise SessionHostError("unknown session", "close")
        self.closed.add(handle)

    def close_all(self) -> None:
        self.closed.update(self.handles)

    def _live(self, handle: str, operation: str) -> tuple[EnumValue, str]:
        if handle not in self.handles:
            raise SessionHostError("unknown session", operation)
        if handle in self.closed:
            raise SessionHostError("closed session", operation)
        return self.handles[handle]


@dataclass
class _LifecycleHost(_Host):
    one_shot_flags: list[bool] = field(default_factory=list)

    def with_ephemeral(
        self, _agent: EnumValue, _transport: str, action: object, *, one_shot: bool = False
    ) -> object:
        self.one_shot_flags.append(one_shot)
        if not callable(action):
            raise AssertionError("expected callable action")
        return action("lifecycle")


def test_with_ephemeral_session_opens_and_closes_non_lifecycle_hosts() -> None:
    host = _Host()
    agent = agent_value("AgentCommand", command="worker")

    first = with_ephemeral_session(host, agent, "Cli", lambda handle: handle)
    second = with_ephemeral_session(host, agent, "Cli", lambda handle: handle, one_shot=True)

    assert {first, second} == host.closed


def test_with_ephemeral_session_delegates_one_shot_lifecycle_hosts() -> None:
    host = _LifecycleHost()
    agent = agent_value("AgentCommand", command="worker")

    assert (
        with_ephemeral_session(host, agent, "Cli", lambda handle: handle, one_shot=True)
        == "lifecycle"
    )
    assert host.one_shot_flags == [True]


def test_dispatcher_session_host_snapshots_its_default_and_preserves_requests() -> None:
    requests: list[AgentRequest] = []
    host = AgentDispatcherSessionHost(
        lambda request: requests.append(request) or AgentResponse("answer", {"source": "test"})
    )
    agent = agent_value("AgentCommand", command="worker")
    handle = host.open_ephemeral(agent, "Cli")
    request = AgentRequest(agent=agent, prompt="question", attempt=2)

    response = host.ask_request(handle, request)

    assert response == AgentResponse("answer", {"source": "test"})
    assert requests == [request]
    assert host.ask(handle, "another question") == "answer"
    host.close(handle)
    assert host.active_session_count == 0
    assert (
        with_ephemeral_session(
            host, agent, "Cli", lambda handle: host.ask(handle, "one shot"), one_shot=True
        )
        == "answer"
    )
    assert host.active_session_count == 0

    no_dispatcher = AgentDispatcherSessionHost(None)
    unavailable_handle = no_dispatcher.open_ephemeral(agent, "Cli")
    with pytest.raises(SessionAskError) as ask_error:
        no_dispatcher.ask(unavailable_handle, "question")
    assert ask_error.value.cause == "no_dispatcher"
    no_dispatcher.close_all()

    failed_dispatcher = AgentDispatcherSessionHost(
        lambda _request: (_ for _ in ()).throw(
            AgentCallHostError(cause="timeout", exit_code=1, stderr_tail="late", elapsed=2.0)
        )
    )
    failed_handle = failed_dispatcher.open_ephemeral(agent, "Cli")
    with pytest.raises(SessionAskError) as dispatch_error:
        failed_dispatcher.ask(failed_handle, "question")
    assert dispatch_error.value.cause == "timeout"

    default = host.default(agent, "Cli")
    assert host.default(agent_value("AgentCommand", command="other"), "Rpc") == default
    assert host.snapshot(default).agent == agent
    assert host.snapshot(default).transport == "Cli"

    unavailable_operations = (
        lambda: host.open(agent, "Cli"),
        lambda: host.compact("missing"),
        lambda: host.reset("missing"),
        lambda: host.fork("missing"),
        lambda: host.set_name("missing", "name"),
        lambda: host.stats("missing"),
        lambda: host.snapshot("missing"),
        lambda: host.close("missing"),
    )
    for operation in unavailable_operations:
        with pytest.raises(SessionHostError):
            operation()


def _run(source: str, host: _Host) -> RunResult:
    return PipelineDriver(session_host=host).run(source)


def test_open_ask_copy_and_lifecycle_operations_reach_their_session() -> None:
    host = _Host()
    result = _run(
        "program def main() -> unit =\n"
        '  let first = Session::open(AgentCommand("one"))\n'
        '  let second = Session::open(AgentCommand("two"))\n'
        "  let alias = copy(first)\n"
        '  first.ask("first")\n'
        '  second.ask("second")\n'
        '  alias.compact("keep facts")\n'
        "  first.reset()\n"
        "  let child = first.fork()\n"
        '  child.set-name("child")\n'
        "  let stats = child.stats()\n"
        "  if stats.input-tokens == 3 => child.close()\n",
        host,
    )

    assert result.ok
    assert host.prompts == {"s1": ["first"], "s2": ["second"], "s3": []}
    assert host.operations == [
        ("s1", "open", ""),
        ("s2", "open", ""),
        ("s1", "compact", "keep facts"),
        ("s1", "reset", ""),
        ("s3", "open", ""),
        ("s1", "fork", "s3"),
        ("s3", "set-name", "child"),
    ]
    assert host.closed == {"s1", "s2", "s3"}


def test_free_ask_uses_the_default_session_and_snapshots_its_agent() -> None:
    host = _Host()
    result = _run(
        "import std/config\n"
        "program def main() -> unit =\n"
        '  std/config::default-agent := AgentCommand("first")\n'
        "  let direct = Session::default()\n"
        '  let first: text = ask("one")\n'
        '  std/config::default-agent := AgentCommand("second")\n'
        "  let second: text = ask! two\n"
        '  direct.ask("three")\n',
        host,
    )

    assert result.ok
    assert list(host.handles) == ["s1"]
    assert host.handles["s1"][0].fields["command"].value == "first"
    assert host.prompts["s1"] == ["one", "two", "three"]


def test_default_session_stays_snapshotted_while_ask_request_reads_the_live_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = _Host()
    result = _run(
        "import std/config\n"
        "program def main() -> unit =\n"
        '  std/config::default-agent := AgentCommand("first")\n'
        '  let first: text = ask("one")\n'
        '  std/config::default-agent := AgentCommand("second")\n'
        '  let request = ask-request("inspect")\n'
        '  let second: text = ask("two")\n'
        '  print (request.agent == AgentCommand("second"))\n',
        host,
    )

    assert result.ok
    assert host.handles["s1"][0].fields["command"].value == "first"
    assert host.prompts["s1"] == ["one", "two"]
    assert capsys.readouterr().out == "true\n"


def test_free_ask_retries_in_the_default_session() -> None:
    class RetryingHost(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            super().ask(handle, prompt)
            return ["not a number", "7"][len(self.prompts[handle]) - 1]

    host = RetryingHost()
    result = _run(
        "program def main() -> unit =\n"
        '  let number: int = ask("count", on_parse_error = Retry(n = 1))\n',
        host,
    )

    assert result.ok
    assert list(host.handles) == ["s1"]
    assert host.prompts["s1"][0].startswith("count")
    assert "Validation errors:" in host.prompts["s1"][1]


def test_default_session_snapshots_agent_and_closed_use_is_catchable() -> None:
    host = _Host()
    result = _run(
        "import std/config\n"
        "program def main() -> unit =\n"
        '  std/config::default-agent := AgentCommand("first")\n'
        "  let first = Session::default()\n"
        '  std/config::default-agent := AgentCommand("second")\n'
        "  let second = Session::default()\n"
        '  first.ask("one")\n'
        "  second.close()\n"
        "  second.close()\n"
        "  try\n"
        '    second.ask("no")\n'
        "  catch SessionError =>\n"
        "    ()\n",
        host,
    )

    assert result.ok
    agent, transport = host.handles["s1"]
    assert agent.fields["command"].value == "first"
    assert transport == "Cli"
    assert host.prompts["s1"] == ["one"]


def test_session_ask_retries_with_corrective_follow_ups_and_maps_transport_errors() -> None:
    class RetryingHost(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            super().ask(handle, prompt)
            return ["not a number", "7", "conversation retained"][len(self.prompts[handle]) - 1]

    host = RetryingHost()
    result = _run(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentCommand("worker"))\n'
        '  let number: int = session.ask("one chance", on_parse_error = Retry(n = 3))\n'
        '  let later: text = session.ask("what did I ask?")\n'
        "  session.close()\n",
        host,
    )

    assert result.ok
    assert len(host.prompts["s1"]) == 3
    assert host.prompts["s1"][0].startswith("one chance")
    assert "one chance" not in host.prompts["s1"][1]
    assert "not a number" not in host.prompts["s1"][1]
    assert "Validation errors:" in host.prompts["s1"][1]
    assert "Return only valid JSON matching the schema." in host.prompts["s1"][1]
    assert host.prompts["s1"][2] == "what did I ask?"

    class AskingFailure(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            del handle, prompt
            raise SessionAskError(
                cause="failed",
                exit_code=2,
                stderr_tail="tail",
                elapsed=1.0,
                call_info=AgentCallInfo(
                    argv=("worker",), prompt_via_stdin=True, elapsed=1.0, exit_code=2
                ),
            )

    failed = _run(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentCommand("worker"))\n'
        "  try\n"
        '    session.ask("fail")\n'
        "  catch AgentCallError =>\n"
        "    ()\n",
        AskingFailure(),
    )
    assert failed.ok

    class NoCallInfoFailure(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            del handle, prompt
            raise SessionAskError(
                cause="failed", exit_code=None, stderr_tail="", elapsed=0.0, call_info=None
            )

    no_call_info = _run(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentCommand("worker"))\n'
        "  try\n"
        '    session.ask("fail")\n'
        "  catch AgentCallError =>\n"
        "    ()\n",
        NoCallInfoFailure(),
    )
    assert no_call_info.ok

    class HostFailure(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            del handle, prompt
            raise SessionHostError("unavailable", "ask")

    host_failure = _run(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentCommand("worker"))\n'
        "  try\n"
        '    session.ask("fail")\n'
        "  catch SessionError =>\n"
        "    ()\n",
        HostFailure(),
    )
    assert host_failure.ok

    class CancelledHost(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            del handle, prompt
            raise AgentCancelled("worker", "declined")

    with pytest.raises(AgentCancelled):
        _run(
            "program def main() -> unit =\n"
            '  let session = Session::open(AgentCommand("worker"))\n'
            '  session.ask("cancel")\n',
            CancelledHost(),
        )

    class InterruptedHost(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            del handle, prompt
            raise KeyboardInterrupt

    with pytest.raises(AgentCancelled) as interrupted:
        _run(
            "program def main() -> unit =\n"
            '  let session = Session::open(AgentCommand("worker"))\n'
            '  session.ask("interrupt")\n',
            InterruptedHost(),
        )
    assert interrupted.value.reason == "interrupted"


def test_session_retry_redacts_schema_invalid_output_from_corrective_feedback() -> None:
    class InvalidTypeRetryHost(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            super().ask(handle, prompt)
            return ['"wrong"', "7"][len(self.prompts[handle]) - 1]

    host = InvalidTypeRetryHost()
    result = _run(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentCommand("worker"))\n'
        '  let number: int = session.ask("parse me", on_parse_error = Retry(n = 1))\n',
        host,
    )

    assert result.ok
    assert "wrong" not in host.prompts["s1"][1]
    assert (
        "Validation errors:\n- The response contains a value with an incorrect type."
        in host.prompts["s1"][1]
    )


def test_session_parse_policies_exhaustion_and_transport_failures_stop_retrying() -> None:
    class AlwaysInvalidHost(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            super().ask(handle, prompt)
            return "invalid"

    for options, expected_attempts in [
        ("", 1),
        (", on_parse_error = Abort", 1),
        (", on_parse_error = Retry(n = 2)", 3),
    ]:
        host = AlwaysInvalidHost()
        result = _run(
            "program def main() -> unit =\n"
            '  let session = Session::open(AgentCommand("worker"))\n'
            "  try\n"
            f'    let number: int = session.ask("parse me"{options})\n'
            "    ()\n"
            "  catch AgentParseError =>\n"
            "    session.close()\n",
            host,
        )

        assert result.ok
        assert len(host.prompts["s1"]) == expected_attempts

    class FailingRetryHost(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            super().ask(handle, prompt)
            if len(self.prompts[handle]) == 1:
                return "invalid"
            raise SessionAskError(
                cause="failed", exit_code=2, stderr_tail="tail", elapsed=1.0, call_info=None
            )

    transport_host = FailingRetryHost()
    transport_result = _run(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentCommand("worker"))\n'
        "  try\n"
        '    let number: int = session.ask("parse me", on_parse_error = Retry(n = 3))\n'
        "    ()\n"
        "  catch AgentCallError =>\n"
        "    session.close()\n",
        transport_host,
    )

    assert transport_result.ok
    assert len(transport_host.prompts["s1"]) == 2


def test_dead_rpc_sessions_map_ask_failures_and_later_operations_to_distinct_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class DeadRpcHost(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            del handle, prompt
            raise SessionAskError(
                cause="timeout",
                exit_code=None,
                stderr_tail="idle timeout",
                elapsed=1.0,
                call_info=None,
            )

        def compact(self, handle: str, instructions: str = "") -> None:
            del handle, instructions
            raise SessionHostError("RPC process has exited", "compact")

    result = _run(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentPi("provider", "model", "high"))\n'
        "  try\n"
        '    session.ask("wait")\n'
        "  catch AgentCallError =>\n"
        "    ()\n"
        "  try\n"
        "    session.compact()\n"
        "  catch SessionError as error =>\n"
        "    print error.operation\n"
        '    if error.message != "" => print "message"\n',
        DeadRpcHost(),
    )

    assert result.ok
    assert capsys.readouterr().out == "compact\nmessage\n"


def test_missing_host_and_lifecycle_errors_become_session_errors() -> None:
    unavailable = PipelineDriver().run(
        "program def main() -> unit =\n"
        "  try\n"
        '    let session = Session::open(AgentCommand("worker"))\n'
        "    session.close()\n"
        "  catch SessionError =>\n"
        "    ()\n"
    )
    assert unavailable.ok

    default_unavailable = PipelineDriver().run(
        "program def main() -> unit =\n"
        "  try\n"
        "    let session = Session::default()\n"
        "    session.close()\n"
        "  catch SessionError =>\n"
        "    ()\n"
    )
    assert default_unavailable.ok

    class FailingDefaultHost(_Host):
        def default(self, agent: EnumValue, transport: str, *, name: str = "") -> str:
            del agent, transport, name
            raise SessionHostError("unavailable", "default")

    default_failed = _run(
        "program def main() -> unit =\n"
        "  try\n"
        "    let session = Session::default()\n"
        "    session.close()\n"
        "  catch SessionError =>\n"
        "    ()\n",
        FailingDefaultHost(),
    )
    assert default_failed.ok

    class FailingOpenHost(_Host):
        def open(self, agent: EnumValue, transport: str, *, name: str = "") -> str:
            del agent, transport, name
            raise SessionHostError("unavailable", "open")

    opening_failed = _run(
        "program def main() -> unit =\n"
        "  try\n"
        '    let session = Session::open(AgentCommand("worker"))\n'
        "    session.close()\n"
        "  catch SessionError =>\n"
        "    ()\n",
        FailingOpenHost(),
    )
    assert opening_failed.ok

    class FailingLifecycleHost(_Host):
        def compact(self, handle: str, instructions: str = "") -> None:
            del handle, instructions
            raise SessionHostError("unsupported", "compact")

    failed = _run(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentCommand("worker"))\n'
        "  try\n"
        "    session.compact()\n"
        "  catch SessionError =>\n"
        "    ()\n",
        FailingLifecycleHost(),
    )
    assert failed.ok


def test_session_cleanup_preserves_program_and_interrupt_errors() -> None:
    class FailingCloseHost(_Host):
        def close_all(self) -> None:
            super().close_all()
            raise RuntimeError("cleanup failed")

    program_failure = _run(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentCommand("worker"))\n'
        '  raise RangeError(message = "primary")\n',
        FailingCloseHost(),
    )
    assert program_failure.error is not None
    assert program_failure.error.type_name == "RangeError"

    class CancelledHost(FailingCloseHost):
        def ask(self, handle: str, prompt: str) -> str:
            del handle, prompt
            raise AgentCancelled("worker", "declined")

    with pytest.raises(AgentCancelled) as cancelled:
        _run(
            "program def main() -> unit =\n"
            '  let session = Session::open(AgentCommand("worker"))\n'
            '  session.ask("cancel")\n',
            CancelledHost(),
        )
    assert cancelled.value.reason == "declined"
    assert any("cleanup failed" in note for note in cancelled.value.__notes__)

    class InterruptedHost(FailingCloseHost):
        def ask(self, handle: str, prompt: str) -> str:
            del handle, prompt
            raise KeyboardInterrupt

    with pytest.raises(AgentCancelled) as interrupted:
        _run(
            "program def main() -> unit =\n"
            '  let session = Session::open(AgentCommand("worker"))\n'
            '  session.ask("interrupt")\n',
            InterruptedHost(),
        )
    assert interrupted.value.reason == "interrupted"
    assert any("cleanup failed" in note for note in interrupted.value.__notes__)


def test_session_cleanup_error_surfaces_after_a_clean_program() -> None:
    class FailingCloseHost(_Host):
        def close_all(self) -> None:
            super().close_all()
            raise RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        _run("program def main() -> unit = ()\n", FailingCloseHost())


def test_default_transport_is_rpc_for_pi_and_host_errors_are_session_errors() -> None:
    host = _Host()
    result = _run(
        "program def main() -> unit =\n"
        '  let pi = Session::open(AgentPi("provider", "model", "think"))\n'
        '  let defaulted = Session::open(AgentCommand("default"), transport = None)\n'
        '  let claude = Session::open(AgentClaude("model", "think"), '
        "transport = Some(SessionTransport::Rpc))\n"
        "  try\n"
        '    claude.ask("bad")\n'
        "  catch SessionError =>\n"
        "    pi.close()\n",
        host,
    )

    assert result.ok
    assert host.handles["s1"][1] == "Rpc"
    assert host.handles["s2"][1] == "Cli"
    assert host.handles["s3"][1] == "Rpc"
