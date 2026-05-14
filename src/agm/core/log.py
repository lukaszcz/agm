"""Shared command output logging helpers."""

from __future__ import annotations

from datetime import datetime
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
    *,
    explicit: bool,
) -> None:
    del explicit
    if log_file is None:
        return
    print(f"Logging to {display_path(log_file)}")
    mkdir(log_file.parent, parents=True, exist_ok=True)
