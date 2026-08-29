"""Reusable fakes and result builders for process-boundary tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agm.core.process import ProcessCaptureResult


def process_result(
    *,
    returncode: int | None = 0,
    stdout: str = "",
    stderr: str = "",
    elapsed: float = 0.01,
    timed_out: bool = False,
    spawn_error: str | None = None,
    spawn_errno: int | None = None,
) -> ProcessCaptureResult:
    """Build a process result with normal-completion defaults."""
    return ProcessCaptureResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed=elapsed,
        timed_out=timed_out,
        spawn_error=spawn_error,
        spawn_errno=spawn_errno,
    )


@dataclass
class FakeShell:
    """Fake ``sh -c`` boundary, in one of two modes.

    With *responses*, each incoming command must equal the next expected
    ``command`` and yields that spec's scripted result; :meth:`assert_complete`
    then checks every response was consumed (an empty list therefore asserts
    that no command ran).  With *responses* left ``None``, every command
    succeeds with *stdout* and nothing is asserted about which commands ran;
    such a fake has nothing to complete, so :meth:`assert_complete` rejects the
    call outright rather than passing vacuously.
    """

    responses: Sequence[Mapping[str, Any]] | None = None
    stdout: str = ""
    commands: list[str] = field(default_factory=list)

    def __call__(
        self,
        args: list[str],
        *,
        idle_timeout: float | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        isolate_process_group: bool = False,
    ) -> ProcessCaptureResult:
        del isolate_process_group
        assert args[:2] == ["sh", "-c"]
        command = args[2]
        index = len(self.commands)
        self.commands.append(command)
        if self.responses is None:
            return process_result(stdout=self.stdout)
        assert index < len(self.responses), f"unexpected shell command: {command!r}"
        spec = self.responses[index]
        assert command == spec["command"], (
            f"shell command {index}: expected {spec['command']!r}, got {command!r}"
        )
        if "env" in spec:
            assert env == spec["env"]
        if "cwd" in spec:
            assert cwd == (None if spec["cwd"] is None else Path(spec["cwd"]))
        if "idle_timeout" in spec:
            assert idle_timeout == spec["idle_timeout"]
        return process_result(
            returncode=spec.get("returncode", 0),
            stdout=spec.get("stdout", ""),
            stderr=spec.get("stderr", ""),
            timed_out=spec.get("timed_out", False),
            spawn_error=spec.get("spawn_error"),
            spawn_errno=spec.get("spawn_errno"),
        )

    def assert_complete(self) -> None:
        """Assert every scripted response was consumed.

        An unscripted fake (``responses=None``) accepts any command, so there
        is nothing for this to check: the call is a test bug and fails loudly
        instead of passing for free.
        """
        assert self.responses is not None, (
            "assert_complete() needs a scripted FakeShell; this one accepts any command"
        )
        assert len(self.commands) == len(self.responses), (
            f"expected {len(self.responses)} shell commands, got {len(self.commands)}"
        )


# ---------------------------------------------------------------------------
# Agent boundary
#
# Every loop, review, revise and refine agent call is spawned through the
# ``run_capture`` subprocess helper.  Faking it there — rather than stubbing
# the AGM functions above it — leaves runner resolution, command
# interpolation, prompt rendering, task selection, output streaming and
# temp-file cleanup running for real.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgentCall:
    """One agent invocation, as it reached the subprocess boundary."""

    argv: list[str]
    prompt: str
    env: dict[str, str]

    @property
    def runner(self) -> list[str]:
        """The runner argv, without the appended ``@<prompt file>`` argument."""
        return self.argv[:-1]

    @property
    def prompt_file(self) -> Path:
        """The prompt file this invocation attached."""
        return Path(self.argv[-1][1:])


@dataclass(frozen=True, slots=True)
class AgentReply:
    """What a faked agent process writes and exits with."""

    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


@dataclass(frozen=True, slots=True)
class AgentTimeout:
    """An agent process killed for sitting idle."""

    message: str = "Idle timeout (1.0s) exceeded, process terminated.\n"


Reply = str | AgentReply | AgentTimeout


@dataclass
class FakeAgent:
    """Answer agent process launches with scripted output.

    Installed over the ``run_capture`` subprocess helper every loop agent call
    is spawned through, so runner resolution, command interpolation, prompt
    rendering, task selection, output streaming and temp-file cleanup all run
    for real — only the process launch is simulated.  *respond* maps each call
    to its reply and may raise to simulate a spawn failure or an interrupt.
    """

    respond: Callable[[AgentCall], Reply]
    calls: list[AgentCall] = field(default_factory=list)

    def __call__(
        self,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        stdout_callback: Callable[[str], None] | None = None,
        stderr_callback: Callable[[str], None] | None = None,
        timeout_callback: Callable[[str], None] | None = None,
        **_kwargs: object,
    ) -> tuple[int, str, str]:
        target = cmd[-1]
        assert target.startswith("@"), f"agent command carries no prompt file: {cmd}"
        call = AgentCall(
            argv=list(cmd),
            prompt=Path(target[1:]).read_text(encoding="utf-8"),
            env=dict(env or {}),
        )
        self.calls.append(call)
        reply = self.respond(call)
        if isinstance(reply, AgentTimeout):
            if timeout_callback is not None:
                timeout_callback(reply.message)
            # ``run_capture`` reports an idle timeout as SystemExit(124).
            raise SystemExit(124)
        if isinstance(reply, str):
            reply = AgentReply(stdout=reply)
        if reply.stdout and stdout_callback is not None:
            stdout_callback(reply.stdout)
        if reply.stderr and stderr_callback is not None:
            stderr_callback(reply.stderr)
        return reply.returncode, reply.stdout, reply.stderr

    @property
    def runners(self) -> list[str]:
        """The executable each recorded call launched, in order."""
        return [call.argv[0] for call in self.calls]

    @property
    def prompts(self) -> list[str]:
        """The prompt text each recorded call carried, in order."""
        return [call.prompt for call in self.calls]

    def prompts_of(self, executable: str) -> list[str]:
        """The prompt text of every call that launched *executable*."""
        return [call.prompt for call in self.calls if call.argv[0] == executable]


def fake_agent(
    monkeypatch: pytest.MonkeyPatch,
    script: Mapping[str, list[Reply]] | Callable[[AgentCall], Reply] | Reply = "",
) -> FakeAgent:
    """Install a :class:`FakeAgent` over the agent subprocess boundary.

    *script* is either a reply for every call, a callable answering each call,
    or a per-executable queue of replies.  A queue also bounds the loop's
    retries: a call the script does not cover fails the test instead of
    spinning.
    """
    respond: Callable[[AgentCall], Reply]
    if isinstance(script, Mapping):
        queues = {name: list(replies) for name, replies in script.items()}

        def respond(call: AgentCall) -> Reply:
            queue = queues.get(call.argv[0])
            assert queue, f"unscripted agent call: {call.argv}"
            return queue.pop(0)

    elif callable(script):
        respond = script
    else:
        constant = script

        def respond(call: AgentCall) -> Reply:
            return constant

    fake = FakeAgent(respond)
    monkeypatch.setattr("agm.agent.runner.run_capture", fake)
    return fake
