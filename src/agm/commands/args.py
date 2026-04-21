"""Typed CLI argument containers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HelpArgs:
    command: str | None
    help_command: list[str]


@dataclass(slots=True)
class ConfigCopyArgs:
    config_command: str | None
    dirname: str


@dataclass(slots=True)
class WorktreeNewArgs:
    worktrees_dir: str | None
    branch: str


@dataclass(slots=True)
class WorktreeSetupArgs:
    wt_command: str | None


@dataclass(slots=True)
class WorktreeRemoveArgs:
    force: bool
    branch: str


@dataclass(slots=True)
class DepNewArgs:
    branch: str | None
    repo_url: str


@dataclass(slots=True)
class DepRemoveArgs:
    all: bool
    target: str


@dataclass(slots=True)
class DepSwitchArgs:
    dep: str
    branch: str
    create_branch: bool


@dataclass(slots=True)
class OpenArgs:
    detached: bool
    pane_count: str | None
    parent: str | None
    branch: str


@dataclass(slots=True)
class CloseArgs:
    branch: str


@dataclass(slots=True)
class InitArgs:
    positional: list[str]
    branch: str | None
    embedded: bool
    workspace: bool


@dataclass(slots=True)
class RunArgs:
    run_command: list[str]
    no_patch: bool
    settings_file: str | None


@dataclass(slots=True)
class LoopArgs:
    command: str | None
    tasks_dir: str | None


@dataclass(slots=True)
class TmuxOpenArgs:
    detach: bool
    pane_count: str | None
    session_name: str | None


@dataclass(slots=True)
class TmuxCloseArgs:
    session_name: str


@dataclass(slots=True)
class TmuxLayoutArgs:
    pane_count: str
    window_id: str | None
