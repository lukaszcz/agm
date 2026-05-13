"""Shared command output logging helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agm.core.fs import append_text, exists, mkdir, read_text, write_text
from agm.vcs import git as git_helpers

AGENT_FILES_DIRNAME = ".agent-files"
AGENT_FILES_GITIGNORE_ENTRY = ".agent-files"


def default_agent_files_dir() -> Path:
    cwd = Path.cwd()
    root = git_helpers.containing_root(cwd)
    return (root if root is not None else cwd) / AGENT_FILES_DIRNAME


def ensure_agent_files_gitignored(agent_files_dir: Path) -> None:
    if agent_files_dir.name != AGENT_FILES_DIRNAME:
        return
    repo_dir = agent_files_dir.parent
    if not git_helpers.is_git_repo(repo_dir):
        return
    gitignore = repo_dir / ".gitignore"
    if exists(gitignore):
        content = read_text(gitignore, encoding="utf-8")
        if AGENT_FILES_GITIGNORE_ENTRY in content.splitlines():
            return
        suffix = "" if content.endswith("\n") else "\n"
        write_text(
            gitignore,
            f"{content}{suffix}{AGENT_FILES_GITIGNORE_ENTRY}\n",
            encoding="utf-8",
        )
        return
    write_text(gitignore, f"{AGENT_FILES_GITIGNORE_ENTRY}\n", encoding="utf-8")


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
    print(f"Logging to {log_file}")
    ensure_agent_files_gitignored(log_file.parent)
    mkdir(log_file.parent, parents=True, exist_ok=True)
