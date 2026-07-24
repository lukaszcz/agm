"""Reusable fakes and result builders for process-boundary tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

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


class _ResponsePolicy(Protocol):
    def __call__(self, index: int, command: str) -> ProcessCaptureResult: ...

    def assert_complete(self, commands: list[str]) -> None: ...


@dataclass(frozen=True)
class _Scripted:
    responses: Sequence[Mapping[str, Any]]

    def __call__(self, index: int, command: str) -> ProcessCaptureResult:
        assert index < len(self.responses), f"unexpected shell command: {command!r}"
        spec = self.responses[index]
        assert command == spec["command"], (
            f"shell command {index}: expected {spec['command']!r}, got {command!r}"
        )
        return process_result(
            returncode=spec.get("returncode", 0),
            stdout=spec.get("stdout", ""),
            stderr=spec.get("stderr", ""),
            timed_out=spec.get("timed_out", False),
            spawn_error=spec.get("spawn_error"),
            spawn_errno=spec.get("spawn_errno"),
        )

    def assert_complete(self, commands: list[str]) -> None:
        assert len(commands) == len(self.responses), (
            f"expected {len(self.responses)} shell commands, got {len(commands)}"
        )


@dataclass(frozen=True)
class _Constant:
    stdout: str

    def __call__(self, index: int, command: str) -> ProcessCaptureResult:
        del index, command
        return process_result(stdout=self.stdout)

    def assert_complete(self, commands: list[str]) -> None:
        del commands


def scripted(responses: Sequence[Mapping[str, Any]]) -> _ResponsePolicy:
    """Return a policy that replays expected commands and process results."""
    return _Scripted(responses)


def constant(*, stdout: str = "") -> _ResponsePolicy:
    """Return a policy that succeeds with the same output for every command."""
    return _Constant(stdout)


@dataclass
class FakeShell:
    """Fake ``sh -c`` boundary parameterized by a process-result policy."""

    response: _ResponsePolicy
    commands: list[str] = field(default_factory=list)

    def __call__(
        self,
        args: list[str],
        *,
        idle_timeout: float | None = None,
        isolate_process_group: bool = False,
    ) -> ProcessCaptureResult:
        del idle_timeout, isolate_process_group
        assert args[:2] == ["sh", "-c"]
        command = args[2]
        self.commands.append(command)
        return self.response(len(self.commands) - 1, command)

    def assert_complete(self) -> None:
        """Assert that the configured response policy consumed every command."""
        self.response.assert_complete(self.commands)
