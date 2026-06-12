"""Shared command output logging helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from agm.core.fs import append_text, mkdir
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
) -> Path | None:
    if no_log:
        return None
    if log_file is not None:
        resolved = Path(log_file)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        return resolved
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return default_agent_files_dir() / f"{command_name}-{timestamp}.log"


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

    ``Decimal`` is emitted as exact fixed-point text (never via float).
    """
    if isinstance(obj, Decimal):
        return format(obj, "f")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def append_jsonl(path: Path | None, record: Mapping[str, object]) -> None:
    """Append *record* as a single JSONL line to *path*.

    A DSL-agnostic helper (plan §11.1): any command that needs structured
    append-a-record logging can use this rather than writing its own JSONL
    emitter.  ``Decimal`` values are serialized exactly (no float round-trip,
    design §5.1).

    If *path* is ``None`` this is a no-op (logging disabled).
    """
    if path is None:
        return
    obj: dict[str, object] = {k: v for k, v in record.items()}
    line = json.dumps(obj, default=_jsonl_default, ensure_ascii=False) + "\n"
    append_text(path, line, encoding="utf-8")
