"""Typed CLI argument containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class ConfigCopyArgs:
    dirname: str


@dataclass(slots=True)
class ConfigEnvArgs:
    pass


@dataclass(slots=True)
class PkgCheckArgs:
    directory: str | None


@dataclass(slots=True)
class PkgInitArgs:
    directory: str | None
    name: str | None
    version: str


@dataclass(slots=True)
class PkgCreateArgs:
    directory: str | None
    output: str | None


@dataclass(slots=True)
class PkgInstallArgs:
    source: str
    editable: bool
    shadow: bool
    reinstall: bool = False


@dataclass(slots=True)
class PkgUninstallArgs:
    name: str


@dataclass(slots=True)
class PkgSwitchArgs:
    target: str


@dataclass(slots=True)
class PkgListArgs:
    pass


@dataclass(slots=True)
class PkgSyncArgs:
    pass


@dataclass(slots=True)
class PkgInfoArgs:
    name: str


@dataclass(slots=True)
class ConfigUpdateArgs:
    pass


@dataclass(slots=True)
class WorktreeNewArgs:
    worktrees_dir: str | None
    branch: str
    no_fetch: bool = True


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
    no_fetch: bool = True


@dataclass(slots=True)
class OpenArgs:
    detached: bool
    pane_count: str | None
    parent: str | None
    branch: str
    no_fetch: bool = True


@dataclass(slots=True)
class CloseArgs:
    branch: str
    force: bool
    force_delete: bool
    keep_branch: bool
    keep_workspace: bool


@dataclass(slots=True)
class InitArgs:
    positional: list[str]
    branch: str | None
    embedded: bool
    split: bool
    clone: bool
    no_repo_git: bool
    no_config_git: bool
    no_notes_git: bool
    no_git_init: bool


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
    pty: bool | None = None


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


@dataclass(slots=True)
class CheckArgs:
    files: list[str]
    module_paths: list[str] = field(default_factory=list)
    no_stdlib: bool = False


class ExecutionOptionValues(Protocol):
    """Execution options shared by commands that run AgL programs."""

    echo: bool | None
    strict_json: bool | None
    default_agent: str | None
    default_sandbox: str | None
    timeout: str | None
    no_timeout: bool
    trace: bool
    no_trace: bool
    trace_file: str | None
    debug: bool | None
    parse_error_retries: int | None


@dataclass(slots=True)
class ExecArgs:
    file: str | None
    strict_json: bool | None
    no_trace: bool
    trace_file: str | None
    echo: bool | None = None
    argument_tokens: list[str] = field(default_factory=list)
    trace: bool = False
    code: str | None = None
    program: str | None = None
    module_paths: list[str] = field(default_factory=list)
    no_stdlib: bool = False
    # Optional recursion call-depth override (None = no override).
    max_call_depth: int | None = None
    timeout: str | None = None
    no_timeout: bool = False
    # An AgL ``Agent`` literal used to seed std/config::default-agent.
    default_agent: str | None = None
    # An AgL ``AgentSandbox`` literal used to seed std/config::default-sandbox.
    default_sandbox: str | None = None
    # Seeds std/config::debug (None = no override).
    debug: bool | None = None
    # Seeds std/config::parse-error-retries (None = no override).
    parse_error_retries: int | None = None
    # Static artifacts produced only when CLI ambiguity required early discovery.
    pipeline_cache: object | None = field(default=None, repr=False, compare=False)


@dataclass(slots=True)
class ReplArgs:
    strict_json: bool | None
    quiet: bool
    no_trace: bool
    trace_file: str | None
    echo: bool | None = None
    trace: bool = False
    # Optional recursion call-depth override (None = no override).
    max_call_depth: int | None = None
    timeout: str | None = None
    no_timeout: bool = False
    no_stdlib: bool = False
    # An AgL ``Agent`` literal used to seed std/config::default-agent.
    default_agent: str | None = None
    # An AgL ``AgentSandbox`` literal used to seed std/config::default-sandbox.
    default_sandbox: str | None = None
    # Seeds std/config::debug (None = no override).
    debug: bool | None = None
    # Seeds std/config::parse-error-retries (None = no override).
    parse_error_retries: int | None = None
    # Force the plain, non-interactive line front end even on a tty; the
    # auto-detected default (non-tty stdin/stdout, or TERM=dumb) needs no flag.
    plain: bool = False
