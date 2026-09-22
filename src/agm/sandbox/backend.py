"""The sandbox backend protocol: pluggable wrapping of a prepared command.

A backend resolves and wraps a :class:`~agm.sandbox.prepare.SandboxRequest`
into the argv that actually enforces isolation. SRT is the only backend that
ships (``agm.sandbox.srt.SRT_BACKEND``); the registry seam here is where a
future method (landrun, bwrap directly, macOS ``sandbox-exec``) would plug in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from agm.sandbox.request import SandboxRequest


class SandboxUnavailableError(Exception):
    """Raised when a sandbox backend, or a dependency it needs, is unavailable.

    ``detail``, when set, is a second line of guidance (e.g. an install
    command) a CLI caller prints on its own line after the main message.
    """

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


class SandboxSettingsError(Exception):
    """Raised when sandbox settings cannot be resolved or are invalid.

    ``path`` (an explicit settings file that was not found) and
    ``candidates`` (the paths checked when none matched) let a CLI caller
    reproduce ``agm run``'s exact stderr messages by formatting them itself,
    e.g. with `agm.core.path.display_path`. The exception's own message stays
    informative on its own for non-CLI callers -- AgL surfaces it inside
    `ExecError`/`AgentCallError`.
    """

    def __init__(
        self,
        message: str,
        *,
        path: Path | None = None,
        candidates: tuple[Path, ...] = (),
    ) -> None:
        super().__init__(message)
        self.path = path
        self.candidates = candidates


@dataclass(frozen=True, slots=True)
class ResolvedSettings:
    """A backend's selected settings and the artifacts preparing it created."""

    path: Path
    temp_files: tuple[Path, ...] = ()
    tracked_artifacts: tuple[Path, ...] = ()


class SandboxBackend(Protocol):
    """One sandboxing method: availability, settings resolution, and argv wrapping."""

    def require_available(self, env: dict[str, str]) -> None:
        """Raise :class:`SandboxUnavailableError` if this backend cannot run."""

    def settings_candidates(self, request: SandboxRequest) -> list[Path]:
        """Return the settings paths *request* would consider, in order.

        Returned whether or not each candidate exists; used both by
        `resolve_settings` and by callers that want the source/candidate list
        (e.g. for dry-run detail lines) without resolving.
        """

    def resolve_settings(self, request: SandboxRequest) -> ResolvedSettings:
        """Select (and merge, if needed) the settings for *request*."""

    def wrap(self, request: SandboxRequest, resolved: ResolvedSettings) -> list[str]:
        """Return the wrapper argv placed in front of *request*'s command."""

    def dry_run_wrap(self, request: SandboxRequest) -> list[str]:
        """Return the wrapper argv with a placeholder in place of a settings path.

        Performs no filesystem resolution: used when a caller wants the
        labeled dry-run command without reading, merging, or writing any
        settings file.
        """

    def prepare_env(self, request: SandboxRequest, env: dict[str, str]) -> dict[str, str]:
        """Return *env* adjusted for running under this backend."""


def default_backend() -> SandboxBackend:
    """Return the sandbox backend AGM ships: SRT.

    The import is local because this is a registry lookup, not a cycle
    workaround: `srt.py` depends on this module and on `request.py`, never
    the reverse, so the indirection exists only to keep backend selection a
    single seam for future methods (landrun, bwrap directly, ...).
    """

    from agm.sandbox.srt import SRT_BACKEND

    return SRT_BACKEND
