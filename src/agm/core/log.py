"""Shared command output logging helpers."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
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


def resolve_log_file(
    *,
    command_name: str,
    no_log: bool,
    log_file: str | None,
    unique: bool = False,
) -> Path | None:
    """Resolve the trace log file path.

    When *unique* is ``True``, a pid-based component is appended to the
    default (timestamp-based) filename so that two concurrent invocations in
    the same second produce different paths instead of colliding (F6, §11.1).
    *unique* has no effect when *log_file* is provided explicitly — the caller
    already owns the path.  loop/review behavior is unchanged unless they opt in
    (they pass ``unique=False``, the default).
    """
    if no_log:
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
    no_log: bool,
    log_file: str | None,
) -> Path | None:
    """Resolve and validate the JSONL trace path up front, or return ``None``.

    Resolves via :func:`resolve_log_file` (``unique=True`` to avoid collisions
    on the second-granularity stamp), then creates the parent directory and
    truncates the file to empty to confirm writability and ensure each run
    starts from a clean file.  An unwritable path exits 1 with a clean
    ``Error: ...`` BEFORE any program runs instead of crashing mid-run.
    Returns ``None`` when *no_log* is set; callers that suppress tracing for
    other reasons (e.g. ``--dry-run``) short-circuit before calling.  Shared
    by ``agm exec`` and ``agm repl``.
    """
    log_path = resolve_log_file(
        command_name=command_name, no_log=no_log, log_file=log_file, unique=True
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
