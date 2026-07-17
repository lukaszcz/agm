"""Live host-service reconfiguration for host-consumed ``builtin var`` settings.

The host-consumed engine settings (``runner``, ``log``, ``log-file``) are backed
by live host services — the agent registry's default agent and the trace store.
When a program writes one of these settings (via ``std.config::runner := ...`` or
the equivalent bare form), the interpreter reflects the change into those live
services through a :class:`HostSettingsReconfigurer`.

The concrete reconfiguration steps (how to build a default agent from a runner
command, how to resolve a trace path from the logging decision) are host policy,
supplied as callables in :class:`HostSettingsPolicy` so this runtime module stays
free of command-layer dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agm.agl.runtime.agents import AgentFn, AgentRegistry
    from agm.agl.runtime.trace import TraceStore


@dataclass(frozen=True)
class HostSettingsPolicy:
    """Host policy for reconfiguring live services on a host-consumed write.

    ``build_runner``
        Maps a runner command string to the default agent callable that
        dispatches unnamed agent calls through it.
    ``resolve_trace_path``
        Maps ``(enabled, log_file)`` to the trace file path to write to, or
        ``None`` when logging is off.
    """

    build_runner: Callable[[str], "AgentFn"]
    resolve_trace_path: Callable[[bool, str | None], Path | None]


class HostSettingsReconfigurer:
    """Applies host-consumed ``builtin var`` writes to the live host services."""

    def __init__(
        self,
        *,
        registry: "AgentRegistry",
        trace: "TraceStore",
        policy: HostSettingsPolicy,
    ) -> None:
        self._registry = registry
        self._trace = trace
        self._policy = policy

    def reconfigure_runner(self, command: str) -> None:
        """Point the default agent at a newly built runner-backed callable."""
        self._registry.set_default_agent(self._policy.build_runner(command))

    def reconfigure_trace(self, *, enabled: bool, log_file: str | None) -> None:
        """Repoint the best-effort trace store at the path implied by the settings."""
        try:
            path = self._policy.resolve_trace_path(enabled, log_file)
        except OSError as exc:
            self._trace.disable(exc)
            return
        self._trace.activate(path)
