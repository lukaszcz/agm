"""Live reconfiguration for reconfigurable host-consumed ``builtin var`` settings.

Host-consumed trace settings are backed by the trace store.

Trace-path resolution is host policy, supplied as a callable in
:class:`HostSettingsPolicy` so this runtime module stays free of command-layer
dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agm.agl.runtime.trace import TraceStore


@dataclass(frozen=True)
class HostSettingsPolicy:
    """Host policy for reconfiguring live services on a host-consumed write.

    ``resolve_trace_path``
        Maps ``(enabled, trace_file)`` to the trace file path to write to, or
        ``None`` when tracing is off.
    """

    resolve_trace_path: Callable[[bool, str | None], Path | None]


class HostSettingsReconfigurer:
    """Applies reconfigurable host-consumed writes to live host services."""

    def __init__(
        self,
        *,
        trace: "TraceStore",
        policy: HostSettingsPolicy,
    ) -> None:
        self._trace = trace
        self._policy = policy

    def reconfigure_trace(self, *, enabled: bool, trace_file: str | None) -> None:
        """Repoint the best-effort trace store at the path implied by the settings."""
        try:
            path = self._policy.resolve_trace_path(enabled, trace_file)
        except OSError as exc:
            self._trace.disable(exc)
            return
        self._trace.activate(path)
