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
class ConfigEnvArgs:
    pass


@dataclass(slots=True)
class ConfigUpdateArgs:
    pass


@dataclass(slots=True)
class WorktreeNewArgs:
    worktrees_dir: str | None
    branch: str


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
    clone: bool


@dataclass(slots=True)
class RunArgs:
    run_command: list[str]
    no_sandbox: bool
    no_patch: bool
    memory: str | None
    swap: str | None
    no_memory_limit: bool
    no_swap_limit: bool
    settings_file: str | None


@dataclass(slots=True)
class LoopArgs:
    command_name: str | None
    runner: str | None
    runner_args: list[str]
    selector: str | None
    no_selector: bool
    tasks_dir: str | None
    no_log: bool
    log_file: str | None
    prompt: str | None
    prompt_file: str | None
    selector_prompt: str | None
    selector_prompt_file: str | None
    timeout: float | None


@dataclass(slots=True)
class LoopNextArgs:
    command_name: str | None
    runner: str | None
    runner_args: list[str]
    selector: str | None
    no_selector: bool
    tasks_dir: str | None
    prompt: str | None
    prompt_file: str | None
    selector_prompt: str | None
    selector_prompt_file: str | None
    timeout: float | None


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
