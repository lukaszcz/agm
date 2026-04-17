"""Typed views over parsed CLI arguments."""

from __future__ import annotations

from typing import Protocol


class HelpArgs(Protocol):
    command: str | None
    help_command: list[str]


class ConfigCopyArgs(Protocol):
    config_command: str | None
    project_dir: str | None
    dirname: str


class WorktreeNewArgs(Protocol):
    worktrees_dir: str | None
    branch: str


class WorktreeSetupArgs(Protocol):
    wt_command: str | None


class WorktreeRemoveArgs(Protocol):
    force: bool
    branch: str


class DepNewArgs(Protocol):
    branch: str | None
    repo_url: str


class DepRemoveArgs(Protocol):
    all: bool
    target: str


class DepSwitchArgs(Protocol):
    dep: str
    branch: str
    create_branch: bool


class OpenArgs(Protocol):
    detached: bool
    pane_count: str | None
    parent: str | None
    branch: str


class CloseArgs(Protocol):
    branch: str


class InitArgs(Protocol):
    positional: list[str]
    branch: str | None
    embedded: bool
    workspace: bool


class RunArgs(Protocol):
    run_command: list[str]
    no_patch: bool
    settings_file: str | None


class TmuxOpenArgs(Protocol):
    command: str | None
    tmux_command: str | None
    detach: bool
    pane_count: str | None
    session_name: str | None


class TmuxCloseArgs(Protocol):
    command: str | None
    tmux_command: str | None
    session_name: str


class TmuxLayoutArgs(Protocol):
    pane_count: str
    window_id: str | None
