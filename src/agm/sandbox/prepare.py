"""Turn a sandbox request into a ready-to-run, cleanup-tracked command.

:func:`prepare` is the one place that combines resource limits, a pluggable
:class:`~agm.sandbox.backend.SandboxBackend`, and an optional PTY wrapper into
a :class:`~agm.sandbox.request.PreparedSandboxCommand`. It never prints or
exits: failures raise :class:`~agm.sandbox.backend.SandboxUnavailableError` or
:class:`~agm.sandbox.backend.SandboxSettingsError`, leaving presentation
(stderr messages, exit codes, ``--dry-run`` wiring) to callers such as
``agm run``, the agent runner, and ``exec``.
"""

from __future__ import annotations

import re
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from agm.config.general import RunConfig
from agm.core import dry_run
from agm.sandbox.backend import (
    SandboxBackend,
    SandboxSettingsError,
    SandboxUnavailableError,
    default_backend,
)
from agm.sandbox.request import (
    DRY_RUN_SETTINGS_PLACEHOLDER,
    Default,
    DefaultLimit,
    LimitSpec,
    PreparedSandboxCommand,
    SandboxRequest,
    SandboxSpec,
)

__all__ = [
    "DEFAULT_MEMORY_LIMIT",
    "DEFAULT_SWAP_LIMIT",
    "Default",
    "DefaultLimit",
    "LimitSpec",
    "PreparedSandboxCommand",
    "ResolvedLimits",
    "SandboxContext",
    "SandboxRequest",
    "SandboxRun",
    "SandboxSpec",
    "dry_run_argv",
    "prepare",
    "print_dry_run",
    "resolve_limits",
    "settings_source",
]

DEFAULT_MEMORY_LIMIT = "32G"
DEFAULT_SWAP_LIMIT = "0"


@dataclass(frozen=True, slots=True)
class SandboxContext:
    """Ambient inputs a `SandboxRequest` needs that a caller does not otherwise carry.

    Built once by a host from its own config context and reused across the
    sandboxed calls it makes (agent runs, `exec`).
    """

    home: Path
    proj_dir: Path | None
    cwd: Path
    run_config: RunConfig

    def prepare(
        self,
        command: list[str],
        spec: SandboxSpec,
        *,
        env: Mapping[str, str],
        pty: bool = False,
        alias_name: str | None = None,
    ) -> PreparedSandboxCommand:
        """Build a `SandboxRequest` from this context and prepare it.

        The one place a caller's `command`/`spec` maps onto this context's
        ambient fields and `run_config` threads through to `prepare()`, so a
        caller never hand-maps `SandboxRequest`'s fields itself.
        """
        request = SandboxRequest(
            command=command,
            cwd=self.cwd,
            env=dict(env),
            home=self.home,
            proj_dir=self.proj_dir,
            spec=spec,
            alias_name=alias_name,
            pty=pty,
        )
        return prepare(request, run_config=self.run_config)


@dataclass(frozen=True, slots=True)
class SandboxRun:
    """A sandbox spec paired with the context that prepares it.

    Replaces two independently-optional parameters (a spec and a context)
    with one value, so a caller cannot supply one without the other.
    """

    spec: SandboxSpec
    context: SandboxContext


# The delegated cgroup lets a resource-limited scope's children join
# `systemd-run`'s scope cgroup so their memory/swap usage is accounted
# together, rather than each landing in its own leaf cgroup unmanaged.
_SYSTEMD_DELEGATED_CGROUP_BOOTSTRAP = (
    "CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup); "
    'mkdir -p "${CG}/init"; '
    'echo $$ > "${CG}/init/cgroup.procs"; '
    'echo "+memory" > "${CG}/cgroup.subtree_control"; '
    'export SANDBOX_CGROUP="$CG"; '
    'exec "$@"'
)

# One <number><suffix?> group: suffix is K/M/G/T/P/E (case-sensitive, powers
# of 1024) optionally followed by B, or a bare B.
_SIZE_GROUP_PATTERN = r"\d+(?:\.\d+)?(?:[KMGTPE]B?|B)?"
_SIZE_PATTERN = re.compile(rf"^{_SIZE_GROUP_PATTERN}(?:\s+{_SIZE_GROUP_PATTERN})*$")
_PERCENT_PATTERN = re.compile(r"^\d+(?:\.\d+)?%$")


def _normalize_systemd_limit(limit: str) -> str:
    """Trim *limit* and normalize a case-insensitive ``unlimited`` to ``infinity``."""

    trimmed = limit.strip()
    if trimmed.lower() == "unlimited":
        return "infinity"
    return trimmed


def _validate_limit(limit: str) -> str:
    """Validate and normalize a systemd resource-limit value.

    Accepts, per ``systemd.resource-control(5)`` and systemd's size parsing:
    the literal ``infinity`` (exact case), a case-insensitive ``unlimited``
    (normalized to ``infinity``), a percentage (``N%``/``N.N%``), or one or
    more whitespace-separated ``<number><suffix?>`` groups summed by systemd,
    where *suffix* is one of ``K M G T P E`` (case-sensitive, powers of 1024)
    optionally followed by ``B``, or a bare ``B``. Surrounding whitespace is
    trimmed. Raises :class:`~agm.sandbox.backend.SandboxSettingsError` for
    anything else.
    """

    normalized = _normalize_systemd_limit(limit)
    if normalized == "infinity" or _PERCENT_PATTERN.match(normalized):
        return normalized
    if _SIZE_PATTERN.match(normalized):
        return normalized
    raise SandboxSettingsError(f"invalid resource limit: {limit!r}")


def _resolve_raw_limit(value: LimitSpec, *, configured: str | None, default: str) -> str | None:
    """Resolve *value* against *configured*/*default* without validating or normalizing it.

    The `None`/`Default`/explicit-string sentinel chain, before `_validate_limit`
    normalizes the result (e.g. ``unlimited`` -> ``infinity``).
    """

    if value is None:
        return None
    if isinstance(value, DefaultLimit):
        return configured or default
    return value


@dataclass(frozen=True, slots=True)
class ResolvedLimits:
    """A sandbox spec's memory/swap limits, both raw and validated.

    ``memory_raw``/``swap_raw`` are the pre-validation values, for a caller
    that displays what is in effect (e.g. ``agm run --dry-run``).
    ``memory``/``swap`` are validated and normalized (e.g. ``unlimited`` ->
    ``infinity``), the values `prepare()` places in the argv.
    """

    memory_raw: str | None
    swap_raw: str | None
    memory: str | None
    swap: str | None


def resolve_limits(spec: SandboxSpec, run_config: RunConfig) -> ResolvedLimits:
    """Resolve *spec*'s memory/swap limits against *run_config*.

    The one place this resolution happens: `prepare()` calls it internally to
    build the argv, and a caller (`agm run --dry-run`) calls it directly to
    render the same values without duplicating the policy. Raises
    :class:`~agm.sandbox.backend.SandboxSettingsError` for an invalid limit.
    """

    memory_raw = _resolve_raw_limit(
        spec.memory,
        configured=run_config.memory_limit_for(spec.profile_name),
        default=DEFAULT_MEMORY_LIMIT,
    )
    swap_raw = _resolve_raw_limit(
        spec.swap,
        configured=run_config.swap_limit_for(spec.profile_name),
        default=DEFAULT_SWAP_LIMIT,
    )
    return ResolvedLimits(
        memory_raw=memory_raw,
        swap_raw=swap_raw,
        memory=_validate_limit(memory_raw) if memory_raw is not None else None,
        swap=_validate_limit(swap_raw) if swap_raw is not None else None,
    )


def _systemd_run_prefix(*, memory_limit: str | None, swap_limit: str | None) -> list[str]:
    prefix = ["systemd-run", "--user", "--scope", "-q"]
    if memory_limit is not None:
        prefix.extend(["-p", f"MemoryMax={memory_limit}"])
    if swap_limit is not None:
        prefix.extend(["-p", f"MemorySwapMax={swap_limit}"])
    prefix.extend(["-p", "Delegate=yes"])
    return prefix


def _systemd_scope_name() -> str:
    return f"agm-run-{uuid4().hex}.scope"


def _resource_limit_run_context(
    env: dict[str, str], memory_limit: str | None, swap_limit: str | None
) -> tuple[list[str], list[str] | None]:
    if memory_limit is None and swap_limit is None:
        return [], None
    if shutil.which("systemd-run", path=env.get("PATH")) is None:
        raise SandboxUnavailableError("systemd-run is not installed or not in PATH.")
    scope_name = _systemd_scope_name()
    return (
        [
            *_systemd_run_prefix(memory_limit=memory_limit, swap_limit=swap_limit),
            "--unit",
            scope_name,
            "--",
            "bash",
            "-c",
            _SYSTEMD_DELEGATED_CGROUP_BOOTSTRAP,
            "--",
        ],
        # --no-block: hand the teardown to systemd and return immediately, so
        # the stop job completes on its own even if this process is killed
        # before the scope's members have finished dying.
        ["systemctl", "--user", "--no-block", "stop", scope_name],
    )


def settings_source(spec: SandboxSpec) -> str:
    """Return ``"explicit"`` or ``"merged"``, mirroring `agm run --dry-run`'s label."""

    return "explicit" if spec.settings_file is not None else "merged"


def prepare(
    request: SandboxRequest,
    *,
    run_config: RunConfig,
    backend: SandboxBackend | None = None,
    resolve_settings: bool = True,
) -> PreparedSandboxCommand:
    """Prepare *request* for execution: resource limits, backend wrap, PTY.

    With ``resolve_settings=False`` (for a dry-run caller that has not
    confirmed it will actually run the command), the backend's settings are
    never read, merged, or written: no temp files, no tracked artifacts, and
    the wrapper argv carries `DRY_RUN_SETTINGS_PLACEHOLDER` instead of a real
    settings path. `dry_run_argv`/`print_dry_run` work the same way in either
    mode.

    Raises :class:`~agm.sandbox.backend.SandboxUnavailableError` or
    :class:`~agm.sandbox.backend.SandboxSettingsError` on failure. Never
    prints or exits.
    """

    limits = resolve_limits(request.spec, run_config)
    process_prefix, interrupt_cleanup_cmd = _resource_limit_run_context(
        request.env, limits.memory, limits.swap
    )

    env = dict(request.env)
    wrapper: list[str] = []
    settings_path: Path | None = None
    temp_files: list[Path] = []
    tracked_artifacts: list[Path] = []

    if request.sandboxed:
        active_backend = backend or default_backend()
        active_backend.require_available(env)
        if resolve_settings:
            resolved = active_backend.resolve_settings(request)
            settings_path = resolved.path
            temp_files.extend(resolved.temp_files)
            tracked_artifacts.extend(resolved.tracked_artifacts)
            wrapper = active_backend.wrap(request, resolved)
        else:
            wrapper = active_backend.dry_run_wrap(request)
        env = active_backend.prepare_env(request, env)

    pty_wrapper = [sys.executable, "-m", "agm.sandbox.pty", "--"] if request.pty else []
    argv = [*process_prefix, *wrapper, *pty_wrapper, *request.command]

    return PreparedSandboxCommand(
        argv=argv,
        env=env,
        cwd=request.cwd,
        interrupt_cleanup_cmd=interrupt_cleanup_cmd,
        settings_path=settings_path,
        _temp_files=temp_files,
        _tracked_artifacts=tracked_artifacts,
    )


def dry_run_argv(
    prepared: PreparedSandboxCommand, *, placeholder: str = DRY_RUN_SETTINGS_PLACEHOLDER
) -> list[str]:
    """Return *prepared*'s argv with any settings-file path replaced by *placeholder*.

    A no-op when *prepared* already carries no `settings_path` -- either it
    was not sandboxed, or it came from `prepare(..., resolve_settings=False)`,
    whose argv already carries `DRY_RUN_SETTINGS_PLACEHOLDER` verbatim.
    """

    if prepared.settings_path is None:
        return list(prepared.argv)
    settings_str = str(prepared.settings_path)
    return [placeholder if part == settings_str else part for part in prepared.argv]


def print_dry_run(prepared: PreparedSandboxCommand, *, label: str = "sandbox") -> None:
    """Print *prepared* through the shared dry-run command printer."""

    dry_run.print_labeled_command(label, dry_run_argv(prepared), cwd=prepared.cwd)
