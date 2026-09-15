"""Runtime behavior for host-backed AgL sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import pytest

from agm.agent.spec import (
    AgentClaude,
    AgentCodex,
    AgentCommand,
    AgentPi,
    AgentSpec,
    SessionTransport,
)
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
    SessionAgentError,
    SessionAskError,
    SessionHostError,
    SessionSnapshot,
    SessionStats,
    default_session_transport,
    with_ephemeral_session,
)


@dataclass
class _Host:
    handles: dict[str, tuple[AgentSpec, str]] = field(default_factory=dict)
    prompts: dict[str, list[str]] = field(default_factory=dict)
    operations: list[tuple[str, str, str]] = field(default_factory=list)
    closed: set[str] = field(default_factory=set)
    default_handle: str | None = None

    def open(self, agent: AgentSpec, transport: str, *, name: str = "") -> str:
        handle = f"s{len(self.handles) + 1}"
        self.handles[handle] = (agent, transport)
        self.prompts[handle] = []
        self.operations.append((handle, "open", name))
        return handle

    def open_ephemeral(
        self, agent: AgentSpec, transport: str, *, single_prompt: bool = False
    ) -> str:
        del single_prompt
        return self.open(agent, transport)

    def default(self, agent: AgentSpec, transport: str, *, name: str = "") -> str:
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

    def _live(self, handle: str, operation: str) -> tuple[AgentSpec, str]:
        if handle not in self.handles:
            raise SessionHostError("unknown session", operation)
        if handle in self.closed:
            raise SessionHostError("closed session", operation)
        return self.handles[handle]


@dataclass
class _LifecycleHost(_Host):
    single_prompt_flags: list[bool] = field(default_factory=list)

    def with_ephemeral(
        self, _agent: AgentSpec, _transport: str, action: object, *, single_prompt: bool = False
    ) -> object:
        self.single_prompt_flags.append(single_prompt)
        if not callable(action):
            raise AssertionError("expected callable action")
        return action("lifecycle")


def test_with_ephemeral_session_opens_and_closes_non_lifecycle_hosts() -> None:
    host = _Host()
    agent = AgentCommand(command="worker")

    first = with_ephemeral_session(host, agent, "Cli", lambda handle: handle)
    second = with_ephemeral_session(host, agent, "Cli", lambda handle: handle, single_prompt=True)

    assert {first, second} == host.closed


def test_with_ephemeral_session_delegates_single_prompt_lifecycle_hosts() -> None:
    host = _LifecycleHost()
    agent = AgentCommand(command="worker")

    assert (
        with_ephemeral_session(host, agent, "Cli", lambda handle: handle, single_prompt=True)
        == "lifecycle"
    )
    assert host.single_prompt_flags == [True]


def test_dispatcher_session_host_snapshots_its_default_and_preserves_requests() -> None:
    requests: list[AgentRequest] = []
    host = AgentDispatcherSessionHost(
        lambda request: requests.append(request) or AgentResponse("answer", {"source": "test"})
    )
    agent = AgentCommand(command="worker")
    handle = host.open_ephemeral(agent, "Cli")
    request = AgentRequest(agent=agent, prompt="question", attempt=2)

    response = host.ask_request(handle, request)

    assert response == AgentResponse("answer", {"source": "test"})
    assert requests == [request]
    assert host.ask(handle, "another question") == "answer"
    host.close(handle)
    with pytest.raises(SessionHostError):
        host.ask(handle, "released")

    ephemeral_handles: list[str] = []

    def _single_prompt(ephemeral: str) -> str:
        ephemeral_handles.append(ephemeral)
        return host.ask(ephemeral, "single prompt")

    assert (
        with_ephemeral_session(host, agent, "Cli", _single_prompt, single_prompt=True) == "answer"
    )
    with pytest.raises(SessionHostError):
        host.ask(ephemeral_handles[0], "released")

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
    assert host.default(AgentCommand(command="other"), "Rpc") == default
    assert host.snapshot(default).agent == agent
    assert host.snapshot(default).transport == "Cli"
    host.close(default)
    host.close(default)
    with pytest.raises(SessionHostError):
        host.ask(default, "closed")

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


def test_dispatcher_retry_replays_the_complete_request_context() -> None:
    requests: list[AgentRequest] = []

    def dispatch(request: AgentRequest) -> AgentResponse:
        requests.append(request)
        return AgentResponse(["not a number", "7"][len(requests) - 1])

    result = PipelineDriver(agent_dispatcher=dispatch).run(
        "program def main() -> unit =\n"
        '  let number: int = ask("count", on-parse-error = Retry(n = 1))\n'
    )

    assert result.ok
    assert requests[0].prompt.startswith("count")
    assert requests[1].prompt.startswith("count")
    assert "Return only valid JSON matching the schema." in requests[1].prompt
    assert "not a number" in requests[1].prompt


def _run(source: str, host: _Host) -> RunResult:
    return PipelineDriver(session_host=host).run(source)


def test_session_open_maps_an_undecodable_agent_value_to_a_session_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared ``Agent`` variant with no host spec becomes a catchable ``SessionError``.

    The evaluator decodes the agent value before ever reaching the session
    host (``EffectHandlers._decode_agent_spec``), so a decode failure must
    surface the same way a host lifecycle failure does -- without any host
    ever being called.
    """
    from agm.agent import spec as agent_spec

    catalog = dict(agent_spec.AGENT_SPECS)
    del catalog["AgentCommand"]
    monkeypatch.setattr(agent_spec, "AGENT_SPECS", catalog)

    result = _run(
        'program def main() -> unit =\n  let session = Session::open(AgentCommand("worker"))\n',
        _Host(),
    )

    assert result.error is not None
    assert result.error.type_name == "SessionError"


def test_session_failures_report_the_session_call_location() -> None:
    class FailingHost(_Host):
        def open(self, agent: AgentSpec, transport: str, *, name: str = "") -> str:
            del agent, transport, name
            raise SessionHostError("unavailable", "open")

    result = _run(
        "program def main() -> unit =\n"
        "  let before = 1\n"
        '  let session = Session::open(AgentCommand("worker"))\n',
        FailingHost(),
    )

    assert result.error is not None
    assert result.error.line == 3
    assert result.error.col == 17


def test_session_operation_failures_report_the_operation_location() -> None:
    class FailingCloseHost(_Host):
        def close(self, handle: str) -> None:
            del handle
            raise SessionHostError("unavailable", "close")

    result = _run(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentCommand("worker"))\n'
        "  session.close()\n",
        FailingCloseHost(),
    )

    assert result.error is not None
    assert result.error.line == 3
    assert result.error.col == 3


def test_agent_method_maps_session_agent_errors_to_agent_call_errors() -> None:
    class InvalidAgentHost(_Host):
        def open(self, agent: AgentSpec, transport: str, *, name: str = "") -> str:
            raise SessionAgentError("invalid agent", "open")

    result = _run(
        'program def main() -> unit =\n  let r: text = AgentCommand("bad").ask("prompt")\n  ()',
        InvalidAgentHost(),
    )

    assert result.error is not None
    assert result.error.type_name == "AgentCallError"


def test_free_ask_maps_an_undecodable_agent_value_to_an_agent_call_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A free ``ask`` on a declared ``Agent`` variant with no host spec is a catchable error.

    Decoding happens once, before any ephemeral session opens, so a decode
    failure never reaches the host.
    """
    from agm.agent import spec as agent_spec

    catalog = dict(agent_spec.AGENT_SPECS)
    del catalog["AgentCommand"]
    monkeypatch.setattr(agent_spec, "AGENT_SPECS", catalog)

    result = _run(
        'program def main() -> unit =\n  let r: text = AgentCommand("bad").ask("prompt")\n  ()',
        _Host(),
    )

    assert result.error is not None
    assert result.error.type_name == "AgentCallError"


def test_persistent_session_ask_maps_an_undecodable_agent_value_to_a_session_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session's stored agent losing its host spec between calls becomes a ``SessionError``.

    Decoding happens once per ``ask``, from the session's own snapshot -- so a
    registry change after a successful ``Session::open`` still surfaces here.
    """
    from agm.agent import spec as agent_spec

    class DriftingHost(_Host):
        def open(self, agent: AgentSpec, transport: str, *, name: str = "") -> str:
            handle = super().open(agent, transport, name=name)
            catalog = dict(agent_spec.AGENT_SPECS)
            del catalog["AgentClaude"]
            monkeypatch.setattr(agent_spec, "AGENT_SPECS", catalog)
            return handle

    result = _run(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentClaude("sonnet", "medium"))\n'
        '  session.ask("prompt")\n'
        "  ()",
        DriftingHost(),
    )

    assert result.error is not None
    assert result.error.type_name == "SessionError"


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
        "  let second: text = ask$ two\n"
        '  direct.ask("three")\n',
        host,
    )

    assert result.ok
    assert list(host.handles) == ["s1"]
    agent = host.handles["s1"][0]
    assert isinstance(agent, AgentCommand)
    assert agent.command == "first"
    assert host.prompts["s1"] == ["one", "two", "three"]


def test_default_session_stays_snapshotted_while_ask_request_is_agent_independent(
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
        "  print request.prompt\n",
        host,
    )

    assert result.ok
    agent = host.handles["s1"][0]
    assert isinstance(agent, AgentCommand)
    assert agent.command == "first"
    assert host.prompts["s1"] == ["one", "two"]
    assert capsys.readouterr().out == "inspect\n"


def test_free_ask_retries_in_the_default_session() -> None:
    class RetryingHost(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            super().ask(handle, prompt)
            return ["not a number", "7"][len(self.prompts[handle]) - 1]

    host = RetryingHost()
    result = _run(
        "program def main() -> unit =\n"
        '  let number: int = ask("count", on-parse-error = Retry(n = 1))\n',
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
    assert isinstance(agent, AgentCommand)
    assert agent.command == "first"
    assert transport == "Cli"
    assert host.prompts["s1"] == ["one"]


class _AskInterruptedHost(_Host):
    """A host whose every ``ask`` is interrupted."""

    def ask(self, handle: str, prompt: str) -> str:
        del handle, prompt
        raise KeyboardInterrupt


def _ask_program(prompt: str, *, catching: str = "") -> str:
    """A one-session program that asks *prompt*, optionally inside a ``try``."""
    if not catching:
        return (
            "program def main() -> unit =\n"
            '  let session = Session::open(AgentCommand("worker"))\n'
            f'  session.ask("{prompt}")\n'
        )
    return (
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentCommand("worker"))\n'
        "  try\n"
        f'    session.ask("{prompt}")\n'
        f"  catch {catching} =>\n"
        "    ()\n"
    )


def test_session_ask_retries_with_corrective_follow_ups() -> None:
    class RetryingHost(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            super().ask(handle, prompt)
            return ["not a number", "7", "conversation retained"][len(self.prompts[handle]) - 1]

    host = RetryingHost()
    result = _run(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentCommand("worker"))\n'
        '  let number: int = session.ask("one chance", on-parse-error = Retry(n = 3))\n'
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


@pytest.mark.parametrize(
    "call_info",
    [
        pytest.param(
            AgentCallInfo(argv=("worker",), prompt_via_stdin=True, elapsed=1.0, exit_code=2),
            id="with-call-info",
        ),
        pytest.param(None, id="without-call-info"),
    ],
)
def test_session_ask_transport_failure_is_catchable_as_an_agent_call_error(
    call_info: AgentCallInfo | None,
) -> None:
    class AskingFailure(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            del handle, prompt
            raise SessionAskError(
                cause="failed",
                exit_code=2 if call_info is not None else None,
                stderr_tail="tail" if call_info is not None else "",
                elapsed=1.0 if call_info is not None else 0.0,
                call_info=call_info,
            )

    assert _run(_ask_program("fail", catching="AgentCallError"), AskingFailure()).ok


def test_session_ask_host_failure_is_catchable_as_a_session_error() -> None:
    class HostFailure(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            del handle, prompt
            raise SessionHostError("unavailable", "ask")

    assert _run(_ask_program("fail", catching="SessionError"), HostFailure()).ok


def test_session_ask_interrupt_becomes_an_interrupted_cancellation() -> None:
    with pytest.raises(AgentCancelled) as interrupted:
        _run(_ask_program("interrupt"), _AskInterruptedHost())
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
        '  let number: int = session.ask("parse me", on-parse-error = Retry(n = 1))\n',
        host,
    )

    assert result.ok
    assert "wrong" not in host.prompts["s1"][1]
    assert (
        "Validation errors:\n- The response contains a value with an incorrect type."
        in host.prompts["s1"][1]
    )


@pytest.mark.parametrize(
    ("options", "expected_attempts"),
    [
        pytest.param("", 1, id="default"),
        pytest.param(", on-parse-error = Abort", 1, id="abort"),
        pytest.param(", on-parse-error = Retry(n = 2)", 3, id="retry"),
    ],
)
def test_session_parse_policies_stop_retrying_when_exhausted(
    options: str, expected_attempts: int
) -> None:
    class AlwaysInvalidHost(_Host):
        def ask(self, handle: str, prompt: str) -> str:
            super().ask(handle, prompt)
            return "invalid"

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


def test_session_retry_stops_at_a_transport_failure() -> None:
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
        '    let number: int = session.ask("parse me", on-parse-error = Retry(n = 3))\n'
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


_CLOSE_AFTER_OPEN = (
    "program def main() -> unit =\n"
    "  try\n"
    '    let session = Session::open(AgentCommand("worker"))\n'
    "    session.close()\n"
    "  catch SessionError =>\n"
    "    ()\n"
)

_CLOSE_AFTER_DEFAULT = (
    "program def main() -> unit =\n"
    "  try\n"
    "    let session = Session::default()\n"
    "    session.close()\n"
    "  catch SessionError =>\n"
    "    ()\n"
)


@pytest.mark.parametrize(
    "source",
    [pytest.param(_CLOSE_AFTER_OPEN, id="open"), pytest.param(_CLOSE_AFTER_DEFAULT, id="default")],
)
def test_a_missing_session_host_becomes_a_catchable_session_error(source: str) -> None:
    assert PipelineDriver().run(source).ok


def test_a_failing_default_becomes_a_catchable_session_error() -> None:
    class FailingDefaultHost(_Host):
        def default(self, agent: AgentSpec, transport: str, *, name: str = "") -> str:
            del agent, transport, name
            raise SessionHostError("unavailable", "default")

    assert _run(_CLOSE_AFTER_DEFAULT, FailingDefaultHost()).ok


def test_a_failing_open_becomes_a_catchable_session_error() -> None:
    class FailingOpenHost(_Host):
        def open(self, agent: AgentSpec, transport: str, *, name: str = "") -> str:
            del agent, transport, name
            raise SessionHostError("unavailable", "open")

    assert _run(_CLOSE_AFTER_OPEN, FailingOpenHost()).ok


def test_a_failing_lifecycle_operation_becomes_a_catchable_session_error() -> None:
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


class _FailingCloseHost(_Host):
    """A host whose end-of-run session cleanup itself fails."""

    def close_all(self) -> None:
        super().close_all()
        raise RuntimeError("cleanup failed")


class _InterruptedFailingCloseHost(_FailingCloseHost, _AskInterruptedHost):
    """Interrupts every ``ask``, then fails cleanup."""


def test_session_cleanup_failure_preserves_a_program_error() -> None:
    program_failure = _run(
        "program def main() -> unit =\n"
        '  let session = Session::open(AgentCommand("worker"))\n'
        '  raise RangeError(message = "primary")\n',
        _FailingCloseHost(),
    )
    assert program_failure.error is not None
    assert program_failure.error.type_name == "RangeError"


def test_session_cleanup_failure_preserves_a_cancellation() -> None:
    with pytest.raises(AgentCancelled) as cancelled:
        _run(_ask_program("interrupt"), _InterruptedFailingCloseHost())
    assert cancelled.value.reason == "interrupted"
    assert any("cleanup failed" in note for note in cancelled.value.__notes__)


def test_session_cleanup_error_surfaces_after_a_clean_program() -> None:
    with pytest.raises(RuntimeError, match="cleanup failed"):
        _run("program def main() -> unit = ()\n", _FailingCloseHost())


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


def test_default_session_transport_is_owned_by_each_agent_specification() -> None:
    defaults: list[tuple[AgentSpec, SessionTransport]] = [
        (AgentCommand(command="worker"), SessionTransport.CLI),
        (AgentClaude(model="m", thinking="high"), SessionTransport.CLI),
        (AgentCodex(model="m", thinking="high"), SessionTransport.CLI),
        (AgentPi(provider="p", model="m", thinking="high"), SessionTransport.RPC),
    ]
    for agent, expected in defaults:
        assert default_session_transport(agent) == expected
