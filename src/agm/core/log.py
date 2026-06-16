"""Shared command output logging helpers."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agm.core.fs import append_text, mkdir, write_text
from agm.core.path import display_path
from agm.vcs import git as git_helpers

AGENT_FILES_DIRNAME = ".agent-files"


def default_agent_files_dir() -> Path:
    cwd = Path.cwd()
    root = git_helpers.containing_root(cwd)
    return (root if root is not None else cwd) / AGENT_FILES_DIRNAME


@dataclass(frozen=True)
class LogDecision:
    """Resolved logging decision from the CLI/pragma/config precedence chain.

    ``enabled`` is the final on/off state.  ``explicit_path`` is the resolved
    log-file path when the user (or config) provided one explicitly; ``None``
    means use the auto-generated timestamped path inside ``.agent-files/``.
    """

    enabled: bool
    explicit_path: str | None  # None → auto timestamped path when enabled


def resolve_log_decision(
    *,
    cli_no_log: bool,
    cli_log: bool,
    cli_log_file: str | None,
    pragma_log: bool | None,
    pragma_log_file: str | None,
    config_log: bool,
    config_log_file: str | None,
) -> LogDecision:
    """Resolve the final logging decision from three priority layers.

    Precedence (highest first): CLI > pragma > config.

    CLI layer:
      - ``--no-log``        → enabled=False, path=None  (explicit disable)
      - ``--log-file PATH`` → enabled=True,  path=PATH
      - ``--log``           → enabled=True,  path=None
      - (none)              → enabled=None   (unset; fall through)

    Pragma layer (Part B; pass ``None`` until pragmas are wired):
      - ``pragma_log_file`` → path=pragma_log_file; enabled=True when present
      - ``pragma_log``      → enabled per value (True/False/None)

    Config layer:
      - ``config_log=True`` or ``config_log_file`` → enabled=True
      - ``config_log=False`` (default) and no file → fall through as None
      Note: config cannot express an explicit False distinctly from default;
      use CLI ``--no-log`` or a pragma to force off when config has defaults.

    Enabled resolution: first non-None of [cli, pragma, config], default False.
    Path resolution:    first non-None of [cli.path, pragma.path, config.path].
    """
    # --- CLI layer ---
    if cli_no_log:
        cli_enabled: bool | None = False
    elif cli_log_file is not None:
        cli_enabled = True
    elif cli_log:
        cli_enabled = True
    else:
        cli_enabled = None
    cli_path = cli_log_file  # None when --log or --no-log; explicit str otherwise

    # --- Pragma layer ---
    pragma_path = pragma_log_file
    if pragma_log_file is not None:
        pragma_enabled: bool | None = True if pragma_log is None else pragma_log
    else:
        pragma_enabled = pragma_log  # True, False, or None

    # --- Config layer ---
    config_path = config_log_file
    config_enabled: bool | None = True if (config_log or config_log_file) else None

    # --- Resolve enabled ---
    enabled_layers = [cli_enabled, pragma_enabled, config_enabled]
    resolved_enabled = next((v for v in enabled_layers if v is not None), False)

    # --- Resolve path ---
    resolved_path = next(
        (p for p in [cli_path, pragma_path, config_path] if p is not None), None
    )

    return LogDecision(enabled=resolved_enabled, explicit_path=resolved_path)


def prepare_trace_log_from_layers(
    *,
    command_name: str,
    cli_no_log: bool,
    cli_log: bool,
    cli_log_file: str | None,
    config_log: bool,
    config_log_file: str | None,
    pragma_log: bool | None = None,
    pragma_log_file: str | None = None,
) -> Path | None:
    """Resolve the CLI/pragma/config logging decision and prepare the trace file.

    Combines :func:`resolve_log_decision` with :func:`prepare_trace_log` so the
    two commands that support the full ``--log``/``--no-log``/``--log-file`` +
    config precedence chain (``agm exec`` and ``agm repl``) share one call site.
    ``agm exec`` forwards the ``log``/``log_file`` config pragmas via
    ``pragma_log``/``pragma_log_file``; ``agm repl`` rejects pragmas and leaves
    them at their ``None`` defaults.  Callers handle the ``--dry-run``
    short-circuit before calling.
    """
    decision = resolve_log_decision(
        cli_no_log=cli_no_log,
        cli_log=cli_log,
        cli_log_file=cli_log_file,
        pragma_log=pragma_log,
        pragma_log_file=pragma_log_file,
        config_log=config_log,
        config_log_file=config_log_file,
    )
    return prepare_trace_log(
        command_name=command_name,
        enabled=decision.enabled,
        log_file=decision.explicit_path,
    )


def resolve_log_file(
    *,
    command_name: str,
    enabled: bool,
    log_file: str | None,
    unique: bool = False,
) -> Path | None:
    """Resolve the trace log file path from an already-resolved decision.

    When *enabled* is ``False`` returns ``None`` (no trace).  When *log_file*
    is provided, it is used as the explicit path (resolved to absolute if
    relative).  Otherwise an auto timestamped filename is generated under
    ``.agent-files/``.

    When *unique* is ``True``, a pid-based component is appended to the
    default (timestamp-based) filename so that two concurrent invocations in
    the same second produce different paths instead of colliding (F6, §11.1).
    *unique* has no effect when *log_file* is provided explicitly.
    loop/review behavior is unchanged unless they opt in (``unique=False``,
    the default).
    """
    if not enabled:
        return None
    if log_file is not None:
        resolved = Path(log_file)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        return resolved
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if unique:
        pid = os.getpid()
        return default_agent_files_dir() / f"{command_name}-{timestamp}-{pid}.log"
    return default_agent_files_dir() / f"{command_name}-{timestamp}.log"


def prepare_trace_log(
    *,
    command_name: str,
    enabled: bool,
    log_file: str | None,
) -> Path | None:
    """Resolve and validate the JSONL trace path up front, or return ``None``.

    Resolves via :func:`resolve_log_file` (``unique=True`` to avoid collisions
    on the second-granularity stamp), then creates the parent directory and
    truncates the file to empty to confirm writability and ensure each run
    starts from a clean file.  An unwritable path exits 1 with a clean
    ``Error: ...`` BEFORE any program runs instead of crashing mid-run.
    Returns ``None`` when *enabled* is ``False``; callers that suppress tracing
    for other reasons (e.g. ``--dry-run``) short-circuit before calling.
    Shared by ``agm exec`` and ``agm repl``.
    """
    log_path = resolve_log_file(
        command_name=command_name, enabled=enabled, log_file=log_file, unique=True
    )
    if log_path is not None:
        try:
            mkdir(log_path.parent, parents=True, exist_ok=True)
            write_text(log_path, "", encoding="utf-8")
        except OSError as exc:
            print(f"Error: cannot write trace log to {log_path}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    return log_path


def append_log(log_file: Path | None, content: str) -> None:
    if log_file is None or not content:
        return
    append_text(log_file, content, encoding="utf-8")


def prepare_log_file(
    log_file: Path | None,
) -> None:
    if log_file is None:
        return
    print(f"Logging to {display_path(log_file)}")
    mkdir(log_file.parent, parents=True, exist_ok=True)


def _jsonl_default(obj: object) -> str:
    """JSON serializer for types not handled by the stdlib encoder.

    Raises ``TypeError`` for any unsupported type — including ``Decimal``.
    There is a single numeric convention for the DSL, and it lives in
    ``agm.agl.runtime.serialize.dumps_exact`` (unquoted exact fixed-point text).
    Emitting a ``Decimal`` here would quote it as a JSON string and silently
    diverge from that convention, so callers MUST pre-serialize any
    ``Decimal``-bearing values before passing the record in (F2).
    """
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def append_jsonl(path: Path | None, record: Mapping[str, object]) -> None:
    """Append *record* as a single JSONL line to *path*.

    A DSL-agnostic helper (plan §11.1): any command that needs structured
    append-a-record logging can use this rather than writing its own JSONL
    emitter.

    This helper has NO special numeric handling: ``Decimal`` (and any other
    non-stdlib-JSON type) raises ``TypeError``.  The single numeric convention
    lives in the DSL serializer (``agm.agl.runtime.serialize.dumps_exact``);
    callers MUST pre-serialize ``Decimal``-bearing values to exact text before
    calling this (F2).

    If *path* is ``None`` this is a no-op (logging disabled).
    """
    if path is None:
        return
    obj: dict[str, object] = {k: v for k, v in record.items()}
    line = json.dumps(obj, default=_jsonl_default, ensure_ascii=False) + "\n"
    append_text(path, line, encoding="utf-8")
