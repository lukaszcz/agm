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
class SandboxLimits:
    """The sandbox shape statable without naming the command it applies to.

    An AgL ``Sandbox`` record decodes directly to this shape: the profile
    name is decided later, from context the decoded value itself never
    carries (the agent's executable, or ``exec``'s first shell word).
    """

    memory: LimitSpec = Default
    swap: LimitSpec = Default
    settings_file: Path | None = None
    patch: bool = True

    def for_command(self, profile_name: str | None) -> "SandboxSpec":
        """Bind these profile-independent limits to *profile_name* for one command.

        *profile_name* is always the real executable (never a spec-name
        table, never ``sh``, never an alias); ``None`` selects no name, so
        the config layer falls through to the unqualified defaults.
        """
        return SandboxSpec(
            memory=self.memory,
            swap=self.swap,
            settings_file=self.settings_file,
            patch=self.patch,
            profile_name=profile_name,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SandboxSpec(SandboxLimits):
    """The caller-facing sandbox shape: a profile name plus limits and overrides."""

    profile_name: str | None


@dataclass(frozen=True, slots=True)
class SandboxRequest:
    """Everything needed to prepare one sandboxed invocation.

    ``cwd`` and ``config_cwd`` answer two different questions: ``cwd`` is the
    working directory the command runs in; ``config_cwd`` is the directory a
    `.sandbox/<name>.json` settings candidate is searched under. They are the
    same directory for `agm run`, but a caller with its own per-call working
    directory (`exec`) keeps `config_cwd` pinned to its host `SandboxContext`
    so the directory a command operates on can never supply the settings that
    confine it.
    """

    command: list[str]
    cwd: Path
    config_cwd: Path
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
