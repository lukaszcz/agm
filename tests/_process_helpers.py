"""Reusable fakes and result builders for process-boundary tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
    succeeds with *stdout* and nothing is asserted about which commands ran.
    """

    responses: Sequence[Mapping[str, Any]] | None = None
    stdout: str = ""
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
        index = len(self.commands)
        self.commands.append(command)
        if self.responses is None:
            return process_result(stdout=self.stdout)
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

    def assert_complete(self) -> None:
        """Assert every scripted response was consumed (no-op when accepting any)."""
        if self.responses is not None:
            assert len(self.commands) == len(self.responses), (
                f"expected {len(self.responses)} shell commands, got {len(self.commands)}"
            )
