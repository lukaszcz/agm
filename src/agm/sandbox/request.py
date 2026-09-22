"""Plain sandbox data types and cleanup helpers, shared by every sandbox module.

A leaf module: it imports nothing else from `agm.sandbox`, so `prepare.py`,
`backend.py`, and `srt.py` can all depend on it without forming a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Placeholder standing in for a resolved settings path in a dry-run argv, both
# when a real path is redacted after resolution and when resolution is
# skipped entirely (`prepare(..., resolve_settings=False)`). A plain string,
# never a `Path`, so it can never be mistaken for a real settings file.
DRY_RUN_SETTINGS_PLACEHOLDER = "<dry-run-settings>"


class DefaultLimit:
    """Sentinel: fall through to config, then the built-in default."""


Default = DefaultLimit()
LimitSpec = str | DefaultLimit | None


@dataclass(frozen=True, slots=True)
class SandboxSpec:
    """The caller-facing sandbox shape: a profile name plus limits and overrides."""

    profile_name: str | None
    memory: LimitSpec = Default
    swap: LimitSpec = Default
    settings_file: Path | None = None
    patch: bool = True


@dataclass(frozen=True, slots=True)
class SandboxRequest:
    """Everything needed to prepare one sandboxed invocation."""

    command: list[str]
    cwd: Path
    env: dict[str, str]
    home: Path
    proj_dir: Path | None
    spec: SandboxSpec
    # Only ever set by `agm run`: aliases never apply to agent or `exec` argv.
    alias_name: str | None = None
    pty: bool = False
    sandboxed: bool = True


def cleanup_artifacts(temp_files: list[Path], tracked_artifacts: list[Path]) -> None:
    """Remove this library's own temp settings files and empty tracked artifacts.

    Uses plain `pathlib`, not `agm.core.fs`: these are the library's own
    ephemeral preparation artifacts, never a user-visible filesystem effect,
    so they must be removed even when dry-run mode is enabled globally, and
    removing them must never print.
    """

    for temp_file in temp_files:
        try:
            temp_file.unlink()
        except FileNotFoundError:
            pass
    for artifact in tracked_artifacts:
        try:
            if artifact.is_file() and artifact.stat().st_size == 0:
                artifact.unlink()
            elif artifact.is_dir():
                artifact.rmdir()
        except OSError:
            pass


@dataclass(slots=True)
class PreparedSandboxCommand:
    """A ready-to-run command plus its cleanup and interrupt-teardown hooks."""

    argv: list[str]
    env: dict[str, str]
    cwd: Path
    interrupt_cleanup_cmd: list[str] | None
    settings_path: Path | None
    _temp_files: list[Path] = field(default_factory=list)
    _tracked_artifacts: list[Path] = field(default_factory=list)
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        """Remove the temp settings files and empty tracked artifacts this run created.

        Idempotent: a second call is a no-op.
        """

        if self._closed:
            return
        self._closed = True
        cleanup_artifacts(self._temp_files, self._tracked_artifacts)
