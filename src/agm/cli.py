"""AGM command-line interface implemented with Typer."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, TypedDict

import typer

import agm.commands.config.copy as config_copy_command
import agm.commands.config.env as config_env_command
import agm.commands.config.update as config_update_command
import agm.commands.dep.list as dep_list_command
import agm.commands.dep.new as dep_new_command
import agm.commands.dep.remove as dep_remove_command
import agm.commands.dep.switch as dep_switch_command
import agm.commands.init as init_command
import agm.commands.loop.run as loop_command
import agm.commands.loop.run as loop_run_command
import agm.commands.loop.select as loop_select_command
import agm.commands.loop.step as loop_step_command
import agm.commands.refine as refine_command
import agm.commands.review as review_command
import agm.commands.revise as revise_command
import agm.commands.run as run_command
import agm.commands.sync.fetch as sync_fetch_command
import agm.commands.sync.pull as sync_pull_command
import agm.commands.tmux.close as tmux_close_command
import agm.commands.tmux.layout as tmux_layout_command
import agm.commands.tmux.open as tmux_open_command
import agm.commands.workspace.close as workspace_close_command
import agm.commands.workspace.list as workspace_list_command
import agm.commands.workspace.open as workspace_open_command
import agm.commands.workspace.setup as workspace_setup_command
import agm.commands.workspace.shell_regen as workspace_shell_regen_command
import agm.commands.worktree.new as worktree_new_command
import agm.commands.worktree.remove as worktree_remove_command
from agm import completion
from agm import parser as parser_helpers
from agm.cli_dispatch import RegisteredCommandGroup, set_dry_run
from agm.cli_support.args import (
    CheckArgs,
    CloseArgs,
    ConfigCopyArgs,
    ConfigEnvArgs,
    ConfigUpdateArgs,
    DepNewArgs,
    DepRemoveArgs,
    DepSwitchArgs,
    ExecArgs,
    InitArgs,
    LoopArgs,
    LoopCommandArgs,
    OpenArgs,
    PkgCheckArgs,
    PkgCreateArgs,
    PkgInfoArgs,
    PkgInitArgs,
    PkgInstallArgs,
    PkgListArgs,
    PkgUninstallArgs,
    RefineArgs,
    ReplArgs,
    ReviewArgs,
    ReviseArgs,
    RunArgs,
    TmuxCloseArgs,
    TmuxLayoutArgs,
    TmuxOpenArgs,
    WorktreeNewArgs,
    WorktreeRemoveArgs,
)
from agm.command_catalog import COMMAND_OVERVIEW
from agm.config.general import parse_timeout
from agm.parser import (
    exit_with_usage_error,
    print_command_help,
    print_help_for_command_path,
    print_overview,
)

if TYPE_CHECKING:
    from agm.agl.runtime.types import ProgramDeclInfo
    from agm.cli_support.program_discovery import ExecProgramDiscovery

_HELP_TEXTS = parser_helpers._HELP_TEXTS
_HELP_ALIASES = parser_helpers._HELP_ALIASES
_COMMAND_OVERVIEW = COMMAND_OVERVIEW

_BASE_CONTEXT_SETTINGS: dict[str, bool | list[str]] = {"help_option_names": []}
_RUN_CONTEXT_SETTINGS: dict[str, bool | list[str]] = {
    "help_option_names": [],
    "allow_extra_args": True,
    "ignore_unknown_options": True,
}
_LOOP_CONTEXT_SETTINGS: dict[str, bool | list[str]] = {
    **_RUN_CONTEXT_SETTINGS,
    "allow_interspersed_args": False,
}


def _command_path_from_context(ctx: typer.Context) -> list[str]:
    path: list[str] = []
    current: typer.Context | None = ctx
    while current is not None and current.parent is not None:
        if current.info_name is not None:
            path.append(current.info_name)
        current = current.parent
    path.reverse()
    return path


def _print_command_help(command_path: Sequence[str]) -> None:
    """Print one command path's help page, or the overview for the empty path."""
    if command_path:
        print_help_for_command_path(list(command_path))
    else:
        print_overview()


def _print_context_help(ctx: typer.Context, param: object, value: bool) -> None:
    del param
    if not value or ctx.resilient_parsing:
        return
    _print_command_help(_command_path_from_context(ctx))
    raise typer.Exit()


def _group_help(ctx: typer.Context, *command_path: str) -> None:
    """Print a command group's own help when it is invoked with no subcommand.

    The root group has no command path of its own and prints the overview.
    """
    if ctx.invoked_subcommand is None:
        _print_command_help(command_path)
        raise typer.Exit()


def _help_option() -> bool:
    return typer.Option(
        False,
        "-h",
        "--help",
        callback=_print_context_help,
        expose_value=False,
        is_eager=True,
    )


def _dry_run_option() -> bool:
    return typer.Option(
        False,
        "--dry-run",
        callback=set_dry_run,
        expose_value=False,
        is_eager=True,
        help="Print commands and AGM operations without executing them.",
    )


def _missing_arguments(command_path: Sequence[str], names: Sequence[str]) -> NoReturn:
    joined = ", ".join(names)
    exit_with_usage_error(command_path, f"error: the following arguments are required: {joined}")


def _check_log_flags_exclusive(
    command: str, *, no_log: bool, log: bool, log_file: str | None
) -> None:
    """Reject combinations of the mutually exclusive trace-logging flags."""
    if sum([no_log, log, log_file is not None]) > 1:
        exit_with_usage_error(
            [command], "error: --log, --no-log, and --log-file are mutually exclusive"
        )


def _require_value(
    value: str | Path | None,
    *,
    command_path: Sequence[str],
    name: str,
) -> str:
    if value is None:
        _missing_arguments(command_path, [name])
    return str(value)


def _loop_option_value(
    args: list[str],
    index: int,
    *,
    command_path: Sequence[str],
    option: str,
) -> tuple[str, int]:
    next_index = index + 1
    if next_index >= len(args):
        exit_with_usage_error(command_path, f"error: {option} requires a value")
    return args[next_index], next_index + 1


def _reject_option_conflict(
    command_path: Sequence[str], first: str, second: str, *, conflicting: bool
) -> None:
    """Reject a command line that gave two options which exclude each other."""
    if conflicting:
        exit_with_usage_error(command_path, f"error: {first} and {second} are mutually exclusive")


#: Loop options that take a value, in every loop command's vocabulary.
_LOOP_VALUE_OPTIONS = (
    "--runner",
    "--selector",
    "--tasks-dir",
    "--prompt",
    "--prompt-file",
    "--selector-prompt",
    "--selector-prompt-file",
    "--extra-prompt",
    "--extra-prompt-file",
    "--extra-selector-prompt",
    "--extra-selector-prompt-file",
)
#: Loop options that stand alone, in every loop command's vocabulary.
_LOOP_FLAG_OPTIONS = ("--no-selector",)
#: Loop option pairs that exclude each other, in the order conflicts are reported.
_LOOP_EXCLUSIVE_OPTIONS = (
    ("--selector", "--no-selector"),
    ("--prompt", "--prompt-file"),
    ("--selector-prompt", "--selector-prompt-file"),
    ("--extra-prompt", "--extra-prompt-file"),
    ("--extra-selector-prompt", "--extra-selector-prompt-file"),
)


@dataclass(frozen=True, slots=True)
class _LoopOptions:
    """The options a loop command line carries, and the operands that follow them."""

    values: Mapping[str, str]
    flags: frozenset[str]
    timeout: float | None
    operands: tuple[str, ...]

    def value(self, option: str) -> str | None:
        return self.values.get(option)

    def flag(self, option: str) -> bool:
        return option in self.flags

    def given(self, option: str) -> bool:
        return option in self.values or self.flag(option)


def _parse_loop_options(
    raw_args: list[str],
    *,
    command_path: Sequence[str],
    value_options: Sequence[str],
    flag_options: Sequence[str],
) -> _LoopOptions:
    """Read a loop command's leading options, stopping at its first operand.

    ``--``, and every token the command has no option for, ends the options and
    begins the operands.
    """
    values: dict[str, str] = {}
    flags: set[str] = set()
    timeout: float | None = None
    index = 0
    while index < len(raw_args):
        token = raw_args[index]
        if token in value_options:
            values[token], index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
        elif token in flag_options:
            flags.add(token)
            index += 1
        elif token == "--timeout":
            timeout_text, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            try:
                timeout = parse_timeout(timeout_text)
            except ValueError as exc:
                exit_with_usage_error(command_path, f"error: {exc}")
        else:
            break

    options = _LoopOptions(
        values=values, flags=frozenset(flags), timeout=timeout, operands=tuple(raw_args[index:])
    )
    for first, second in _LOOP_EXCLUSIVE_OPTIONS:
        _reject_option_conflict(
            command_path, first, second, conflicting=options.given(first) and options.given(second)
        )
    return options


class _LoopFields(TypedDict):
    """The argument fields every loop command fills from its options and operands."""

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
    extra_prompt: str | None
    extra_prompt_file: str | None
    extra_selector_prompt: str | None
    extra_selector_prompt_file: str | None
    timeout: float | None


def _loop_fields(options: _LoopOptions) -> _LoopFields:
    """Project parsed loop options onto the argument fields they fill."""
    operands = options.operands
    return _LoopFields(
        command_name=operands[0] if operands else None,
        runner=options.value("--runner"),
        runner_args=run_command.normalize_run_command(list(operands[1:])),
        selector=options.value("--selector"),
        no_selector=options.flag("--no-selector"),
        tasks_dir=options.value("--tasks-dir"),
        prompt=options.value("--prompt"),
        prompt_file=options.value("--prompt-file"),
        selector_prompt=options.value("--selector-prompt"),
        selector_prompt_file=options.value("--selector-prompt-file"),
        extra_prompt=options.value("--extra-prompt"),
        extra_prompt_file=options.value("--extra-prompt-file"),
        extra_selector_prompt=options.value("--extra-selector-prompt"),
        extra_selector_prompt_file=options.value("--extra-selector-prompt-file"),
        timeout=options.timeout,
    )


def _parse_loop_args(
    raw_args: list[str],
    *,
    command_path: Sequence[str],
    command_optional: bool = False,
) -> LoopArgs:
    """Parse a loop command line, which adds the trace-logging options."""
    options = _parse_loop_options(
        raw_args,
        command_path=command_path,
        value_options=(*_LOOP_VALUE_OPTIONS, "--log-file"),
        flag_options=(*_LOOP_FLAG_OPTIONS, "--no-log"),
    )
    if not options.operands and not command_optional:
        print_help_for_command_path(command_path)
        raise typer.Exit()
    _reject_option_conflict(
        command_path,
        "--no-log",
        "--log-file",
        conflicting=options.flag("--no-log") and options.value("--log-file") is not None,
    )
    return LoopArgs(
        no_log=options.flag("--no-log"),
        log_file=options.value("--log-file"),
        **_loop_fields(options),
    )


def _parse_loop_select_args(
    raw_args: list[str], *, command_path: Sequence[str] = ("loop", "select")
) -> LoopCommandArgs:
    """Parse a ``loop select`` command line, which has no trace-logging options."""
    options = _parse_loop_options(
        raw_args,
        command_path=command_path,
        value_options=_LOOP_VALUE_OPTIONS,
        flag_options=_LOOP_FLAG_OPTIONS,
    )
    return LoopCommandArgs(**_loop_fields(options))


def _validate_prompt_options(
    *,
    command_path: Sequence[str],
    prompt: str | None,
    prompt_file: str | None,
    extra_prompt: str | None,
    extra_prompt_file: str | None,
) -> None:
    _reject_option_conflict(
        command_path,
        "--prompt",
        "--prompt-file",
        conflicting=prompt is not None and prompt_file is not None,
    )
    _reject_option_conflict(
        command_path,
        "--extra-prompt",
        "--extra-prompt-file",
        conflicting=extra_prompt is not None and extra_prompt_file is not None,
    )


def _validate_refine_prompt_options(
    *,
    command_path: Sequence[str],
    prompt_name: str,
    prompt: str | None,
    prompt_file: str | None,
    extra_prompt: str | None,
    extra_prompt_file: str | None,
) -> None:
    _reject_option_conflict(
        command_path,
        f"--{prompt_name}-prompt",
        f"--{prompt_name}-prompt-file",
        conflicting=prompt is not None and prompt_file is not None,
    )
    _reject_option_conflict(
        command_path,
        f"--extra-{prompt_name}-prompt",
        f"--extra-{prompt_name}-prompt-file",
        conflicting=extra_prompt is not None and extra_prompt_file is not None,
    )


def _parse_max_steps(value: str | None, *, command_path: Sequence[str], name: str) -> int | None:
    if value is None:
        return None
    if value.strip().lower() == "unlimited":
        return None
    try:
        parsed = int(value)
    except ValueError:
        exit_with_usage_error(
            command_path, f"error: {name} must be a positive integer or 'unlimited'"
        )
    if parsed < 1:
        exit_with_usage_error(command_path, f"error: {name} must be positive")
    return parsed


app = typer.Typer(
    cls=RegisteredCommandGroup,
    context_settings=_BASE_CONTEXT_SETTINGS,
    invoke_without_command=True,
)

config_app = typer.Typer(context_settings=_BASE_CONTEXT_SETTINGS, invoke_without_command=True)
worktree_app = typer.Typer(context_settings=_BASE_CONTEXT_SETTINGS, invoke_without_command=True)
workspace_app = typer.Typer(context_settings=_BASE_CONTEXT_SETTINGS, invoke_without_command=True)
sync_app = typer.Typer(context_settings=_BASE_CONTEXT_SETTINGS, invoke_without_command=True)
dep_app = typer.Typer(context_settings=_BASE_CONTEXT_SETTINGS, invoke_without_command=True)
pkg_app = typer.Typer(context_settings=_BASE_CONTEXT_SETTINGS, invoke_without_command=True)
tmux_app = typer.Typer(context_settings=_BASE_CONTEXT_SETTINGS, invoke_without_command=True)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _group_help(ctx)


@app.command()
def help(
    help_command: list[str] | None = typer.Argument(
        None,
        metavar="command",
        autocompletion=completion.complete_help_path,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    if not help_command:
        print_overview()
        raise typer.Exit()
    try:
        print_help_for_command_path(help_command)
    except ValueError:
        from agm.cli_dispatch import print_registered_command_help

        if not print_registered_command_help(help_command):
            print_command_help(" ".join(help_command))
    raise typer.Exit()


def _run_workspace_open(
    *,
    command_path: Sequence[str],
    target: str | None,
    detached: bool,
    pane_count: str | None,
    parent: str | None,
) -> None:
    workspace_open_command.run(
        OpenArgs(
            detached=detached,
            pane_count=pane_count,
            parent=parent,
            branch=_require_value(target, command_path=command_path, name="target"),
        )
    )


def _run_workspace_close(
    *,
    command_path: Sequence[str],
    branch: str | None,
    force: bool,
    force_delete: bool,
    keep_branch: bool,
    keep_workspace: bool,
) -> None:
    workspace_close_command.run(
        CloseArgs(
            branch=_require_value(
                branch,
                command_path=command_path,
                name="branch",
            ),
            force=force,
            force_delete=force_delete,
            keep_branch=keep_branch or keep_workspace,
            keep_workspace=keep_workspace,
        )
    )


@app.command()
def open(
    target: str | None = typer.Argument(
        None,
        metavar="TARGET",
        autocompletion=completion.complete_open_target,
    ),
    detached: bool = typer.Option(
        False, "-d", "--detach", "--detached", help="Open the session detached."
    ),
    pane_count: str | None = typer.Option(
        None,
        "-n",
        "--num-panes",
        help="Create the session with this many panes.",
        autocompletion=completion.complete_pane_count,
    ),
    parent: str | None = typer.Option(
        None,
        "-p",
        "--parent",
        help="Base a new branch on this workspace.",
        autocompletion=completion.complete_worktree_branch,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _run_workspace_open(
        command_path=["open"],
        target=target,
        detached=detached,
        pane_count=pane_count,
        parent=parent,
    )


@app.command()
def close(
    branch: str | None = typer.Argument(
        None,
        metavar="BRANCH",
        autocompletion=completion.complete_close_branch,
    ),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help=(
            "Force remove the branch workspace's Git worktree (even with untracked files) "
            "and force delete the branch (git branch -D)."
        ),
    ),
    force_delete: bool = typer.Option(
        False,
        "-D",
        help="Force delete the branch (git branch -D) instead of safe delete (git branch -d).",
    ),
    keep_branch: bool = typer.Option(
        False,
        "--keep-branch",
        help="Remove the branch workspace's Git worktree but keep the local branch.",
    ),
    keep_workspace: bool = typer.Option(
        False,
        "--keep-workspace",
        help="Keep the branch workspace and local branch; only close the workspace session.",
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _run_workspace_close(
        command_path=["close"],
        branch=branch,
        force=force,
        force_delete=force_delete,
        keep_branch=keep_branch,
        keep_workspace=keep_workspace,
    )


@config_app.callback(invoke_without_command=True)
def config_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _group_help(ctx, "config")


@config_app.command(name="cp")
def config_cp(
    dirname: Path | None = typer.Argument(
        None,
        metavar="DIRNAME",
        autocompletion=completion.complete_path_argument,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    config_copy_command.run(
        ConfigCopyArgs(
            config_command="cp",
            dirname=_require_value(dirname, command_path=["config", "cp"], name="dirname"),
        )
    )


@config_app.command(name="copy")
def config_copy(
    dirname: Path | None = typer.Argument(
        None,
        metavar="DIRNAME",
        autocompletion=completion.complete_path_argument,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    config_copy_command.run(
        ConfigCopyArgs(
            config_command="copy",
            dirname=_require_value(dirname, command_path=["config", "copy"], name="dirname"),
        )
    )


@config_app.command(name="env")
def config_env(
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    config_env_command.run(ConfigEnvArgs())


@config_app.command(name="update")
def config_update(
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    config_update_command.run(ConfigUpdateArgs())


@workspace_app.callback(invoke_without_command=True)
def workspace_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _group_help(ctx, ctx.info_name or "workspace")


@workspace_app.command(name="open")
def workspace_open(
    target: str | None = typer.Argument(
        None,
        metavar="TARGET",
        autocompletion=completion.complete_open_target,
    ),
    detached: bool = typer.Option(
        False, "-d", "--detach", "--detached", help="Open the session detached."
    ),
    pane_count: str | None = typer.Option(
        None,
        "-n",
        "--num-panes",
        help="Create the session with this many panes.",
        autocompletion=completion.complete_pane_count,
    ),
    parent: str | None = typer.Option(
        None,
        "-p",
        "--parent",
        help="Base a new branch on this workspace.",
        autocompletion=completion.complete_worktree_branch,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _run_workspace_open(
        command_path=["workspace", "open"],
        target=target,
        detached=detached,
        pane_count=pane_count,
        parent=parent,
    )


@workspace_app.command(name="close")
def workspace_close(
    branch: str | None = typer.Argument(
        None,
        metavar="BRANCH",
        autocompletion=completion.complete_close_branch,
    ),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help=(
            "Force remove the branch workspace's Git worktree (even with untracked files) "
            "and force delete the branch (git branch -D)."
        ),
    ),
    force_delete: bool = typer.Option(
        False,
        "-D",
        help="Force delete the branch (git branch -D) instead of safe delete (git branch -d).",
    ),
    keep_branch: bool = typer.Option(
        False,
        "--keep-branch",
        help="Remove the branch workspace's Git worktree but keep the local branch.",
    ),
    keep_workspace: bool = typer.Option(
        False,
        "--keep-workspace",
        help="Keep the branch workspace and local branch; only close the workspace session.",
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _run_workspace_close(
        command_path=["workspace", "close"],
        branch=branch,
        force=force,
        force_delete=force_delete,
        keep_branch=keep_branch,
        keep_workspace=keep_workspace,
    )


@workspace_app.command(name="setup")
def workspace_setup(
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    workspace_setup_command.run()


@workspace_app.command(name="list")
def workspace_list(
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show workspace directories."),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    workspace_list_command.run(verbose=verbose)


@workspace_app.command(name="shell-regen")
def workspace_shell_regen(
    shell_dir: str | None = typer.Argument(
        None, metavar="SHELL_DIR", help="Per-session shell directory to regenerate."
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    workspace_shell_regen_command.run(
        shell_dir=_require_value(
            shell_dir, command_path=["workspace", "shell-regen"], name="shell_dir"
        )
    )


@worktree_app.callback(invoke_without_command=True)
def worktree_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _group_help(ctx, ctx.info_name or "worktree")


@worktree_app.command()
def new(
    branch: str | None = typer.Argument(
        None,
        metavar="BRANCH",
        autocompletion=completion.complete_worktree_branch,
    ),
    worktrees_dir: Path | None = typer.Option(
        None,
        "-d",
        "--dir",
        help="Create the worktree under DIR.",
        autocompletion=completion.complete_path_argument,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    worktree_new_command.run(
        WorktreeNewArgs(
            worktrees_dir=str(worktrees_dir) if worktrees_dir is not None else None,
            branch=_require_value(branch, command_path=["worktree", "new"], name="branch"),
        )
    )


def _print_program_candidates(entry_programs: "tuple[ProgramDeclInfo, ...]") -> None:
    """List the entry programs a reader can select with ``-p``.

    Printed instead of one program's own help when the source declares
    several and none was selected: no single command stands for the
    invocation, so the reader is shown what to select.
    """
    if len(entry_programs) < 2:
        return
    print()
    print("Declared programs (select one with -p to see its own options):")
    for candidate in entry_programs:
        print(f"  {candidate.declaration_path}")


def _exec_print_help(
    discovery: "ExecProgramDiscovery",
    *,
    tokens: "Sequence[str]",
    file: str | None,
    command: str | None,
    program: str | None = None,
) -> bool:
    """Print the help *tokens* request, if they request any; report whether they did.

    The selected program's own command owns ``-h``/``--help``, so Click reads
    the tokens and decides: a ``-h`` supplied as a value-taking option's own
    value is that value, not a help request, and this invocation runs
    normally. When it is a help request, the program's own command help is
    what a reader gets — its usage line, its ``@doc``, and one entry per
    visible parameter. Without a program to build a command from (no source
    selector, a source that does not compile, several entry programs and none
    selected, or a colliding flag projection) ``agm exec``'s own help is
    printed instead, followed by the selectable programs when several were
    discovered.
    """
    from agm.cli_support.program_discovery import unmatched_program_message
    from agm.cli_support.program_options import (
        contains_help_flag,
        exec_program_help,
        program_command_for,
        program_help_requested,
    )

    if not contains_help_flag(tokens):
        return False
    if file is None and command is None:
        print_help_for_command_path(["exec"])
        return True
    selection = discovery.selection(file)
    program_command = program_command_for(selection.selected)
    if not program_help_requested(tokens, program_command):
        return False
    if program_command is None:
        if selection.requested_unmatched:
            # The reader named a program that does not exist. Help and
            # execution agree on that: there is no command to describe, so the
            # request fails exactly as running it would.
            print(
                unmatched_program_message(selection.requested, selection.entry_programs),
                file=sys.stderr,
            )
            raise SystemExit(1)
        print_help_for_command_path(["exec"])
        _print_program_candidates(selection.entry_programs)
        return True
    print(
        exec_program_help(program_command, file=file, program=program),
        end="",
    )
    return True


@app.command(name="exec", context_settings=_RUN_CONTEXT_SETTINGS, cls=completion.ExecCommand)
def exec_cmd(
    tail: list[str] | None = typer.Argument(
        None,
        metavar="FILE",
        autocompletion=completion.complete_agl_file,
    ),
    command: str | None = typer.Option(
        None,
        "-c",
        "--command",
        help="Execute the AgL program given as COMMAND instead of reading from FILE.",
    ),
    program: str | None = typer.Option(
        None,
        "-p",
        "--program",
        metavar="PATH",
        help="Select a program def by its declaration path.",
    ),
    strict_json: bool | None = typer.Option(
        None,
        "--strict-json/--no-strict-json",
        help="Require agents to return exactly one bare JSON value; default is lenient recovery.",
    ),
    max_iters: int | None = typer.Option(
        None,
        "--max-iters",
        help="Positive cap for unbounded loops without an inline bound; off by default.",
    ),
    max_call_depth: int | None = typer.Option(
        None,
        "--max-call-depth",
        help="Override the maximum recursion call depth (CLI > config).",
    ),
    agent: str | None = typer.Option(
        None,
        "--agent",
        metavar="AGL_LITERAL",
        help="Seed the free-ask default session from an AgL Agent literal.",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Write a structured JSONL trace log to PATH. Trace logging is off by default.",
        autocompletion=completion.complete_path_argument,
    ),
    no_log: bool = typer.Option(
        False,
        "--no-log",
        help="Disable trace logging (overrides [exec] log = true in config.toml).",
    ),
    log: bool = typer.Option(
        False,
        "--log",
        help=(
            "Enable trace logging to an auto-named timestamped file under .agent-files/. "
            "Trace logging is off by default; --log, --log-file, or [exec] log = true in "
            "config.toml opt in."
        ),
    ),
    module_paths: list[str] = typer.Option(
        [],
        "-I",
        "--module-path",
        metavar="DIR",
        help=(
            "Add DIR as an additional module search root (repeatable). "
            "Resolved relative to the invocation working directory. "
            "Joins the unordered root set; a module id found in two roots is an ambiguity error."
        ),
        autocompletion=completion.complete_dir_argument,
    ),
    no_stdlib: bool = typer.Option(
        False,
        "--no-stdlib",
        help=(
            "Disable the automatic import std/prelude::* prelude throughout the loaded program "
            "(entry and library modules)."
        ),
    ),
    timeout: str | None = typer.Option(
        None,
        "--timeout",
        help=(
            "Override initial shell-exec and agent idle timeouts (e.g. '30s', '5m', '120').  "
            "Seeds the in-program 'std/config::timeout' setting to Some(VALUE).  "
            "Mutually exclusive with --no-timeout."
        ),
    ),
    no_timeout: bool = typer.Option(
        False,
        "--no-timeout",
        help=(
            "Remove any configured initial shell-exec and agent timeout.  "
            "Seeds the in-program 'std/config::timeout' setting to None.  "
            "Mutually exclusive with --timeout."
        ),
    ),
    no_log_file: bool = typer.Option(
        False,
        "--no-log-file",
        help=(
            "Clears only the in-program log-file binding; a log-file path set in "
            "config or auto-assigned by --log still applies.  Use --no-log to disable "
            "tracing entirely.  Mutually exclusive with --log-file."
        ),
    ),
    _dry_run: bool = _dry_run_option(),
) -> None:
    # ``_RUN_CONTEXT_SETTINGS`` disables Click's built-in ``--help`` interception
    # (``help_option_names: []``) and lets unknown options through, so the whole
    # tail — the FILE argument, the program's own option tokens, and the help
    # flags — arrives here as one catch-all.  ``split_exec_tail`` is the single
    # place that says which token is the FILE and which the program reads.
    from agm.cli_support.program_discovery import ExecProgramDiscovery
    from agm.cli_support.program_options import split_exec_tail

    # One discovery for the whole invocation: the tail split probes candidate
    # FILE tokens with it, and the help surface below then asks about the
    # token it settled on, without paying for a second static pipeline pass.
    discovery = ExecProgramDiscovery(
        command=command,
        requested_program=program,
        module_paths=module_paths,
        no_stdlib=no_stdlib,
    )
    selected = split_exec_tail(
        tail or (),
        program_command_for_file=None if command is not None else discovery.command_for_file,
    )
    file = selected.file
    argument_tokens = list(selected.tokens)
    if _exec_print_help(
        discovery,
        tokens=argument_tokens,
        file=file,
        command=command,
        program=program,
    ):
        raise SystemExit(0)
    del _dry_run
    if command is not None and file is not None:
        exit_with_usage_error(["exec"], "error: argument FILE not allowed with -c/--command")
    if command is None and file is None:
        exit_with_usage_error(["exec"], "error: one of the arguments FILE -c/--command is required")
    _check_log_flags_exclusive("exec", no_log=no_log, log=log, log_file=log_file)
    _reject_option_conflict(
        ["exec"], "--log-file", "--no-log-file", conflicting=log_file is not None and no_log_file
    )
    _reject_option_conflict(
        ["exec"], "--timeout", "--no-timeout", conflicting=timeout is not None and no_timeout
    )
    # Imported lazily: pulls in the AgL DSL (runtime, codec, jsonschema), which
    # would otherwise slow every non-AgL ``agm`` invocation's startup.
    import agm.commands.exec as exec_command

    exec_command.run(
        ExecArgs(
            file=file,
            command=command,
            program=program,
            argument_tokens=argument_tokens,
            strict_json=strict_json,
            max_iters=max_iters,
            max_call_depth=max_call_depth,
            agent=agent,
            no_log=no_log,
            log_file=log_file,
            log=log,
            module_paths=module_paths,
            no_stdlib=no_stdlib,
            timeout=timeout,
            no_timeout=no_timeout,
            no_log_file=no_log_file,
        )
    )


@app.command(name="repl")
def repl_cmd(
    strict_json: bool | None = typer.Option(
        None,
        "--strict-json/--no-strict-json",
        help="Require agents to return exactly one bare JSON value; default is lenient recovery.",
    ),
    max_iters: int | None = typer.Option(
        None,
        "--max-iters",
        help="Positive cap for unbounded loops without an inline bound; off by default.",
    ),
    max_call_depth: int | None = typer.Option(
        None,
        "--max-call-depth",
        help="Override the maximum recursion call depth (CLI > config).",
    ),
    agent: str | None = typer.Option(
        None,
        "--agent",
        metavar="AGL_LITERAL",
        help="Seed the free-ask default session from an AgL Agent literal.",
    ),
    confirm_agents: bool = typer.Option(
        False,
        "--confirm-agents",
        help="Confirm each agent call before dispatching it.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Suppress automatic echoing of entry results.",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Write a structured JSONL trace log to PATH. Trace logging is off by default.",
        autocompletion=completion.complete_path_argument,
    ),
    no_log: bool = typer.Option(
        False,
        "--no-log",
        help="Disable trace logging.",
    ),
    log: bool = typer.Option(
        False,
        "--log",
        help=(
            "Enable trace logging to an auto-named timestamped file under .agent-files/. "
            "Trace logging is off by default; --log, --log-file, or [exec] log = true in "
            "config.toml opt in. A std/config::log write takes effect in the REPL too."
        ),
    ),
    no_stdlib: bool = typer.Option(
        False,
        "--no-stdlib",
        help=(
            "Disable the automatic import std/prelude::* prelude for each loaded REPL program "
            "(entries and library modules)."
        ),
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help=(
            "Force the plain, non-interactive line front end. Auto-detected "
            "otherwise: engaged when stdin/stdout is not a terminal or TERM=dumb."
        ),
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _check_log_flags_exclusive("repl", no_log=no_log, log=log, log_file=log_file)
    # Imported lazily: pulls in the AgL DSL (runtime, repl console), which would
    # otherwise slow every non-AgL ``agm`` invocation's startup.
    import agm.commands.repl as repl_command

    repl_command.run(
        ReplArgs(
            strict_json=strict_json,
            max_iters=max_iters,
            max_call_depth=max_call_depth,
            agent=agent,
            confirm_agents=confirm_agents,
            quiet=quiet,
            no_log=no_log,
            log_file=log_file,
            log=log,
            no_stdlib=no_stdlib,
            plain=plain,
        )
    )


@app.command(name="check")
def check_cmd(
    file: list[str] | None = typer.Argument(
        None,
        metavar="FILE...",
        autocompletion=completion.complete_agl_file,
    ),
    module_paths: list[str] = typer.Option(
        [],
        "-I",
        "--module-path",
        metavar="DIR",
        help=(
            "Add DIR as an additional module search root (repeatable). "
            "Resolved relative to the invocation working directory. "
            "Joins the unordered root set; a module id found in two roots is an ambiguity error."
        ),
        autocompletion=completion.complete_dir_argument,
    ),
    no_stdlib: bool = typer.Option(
        False,
        "--no-stdlib",
        help=(
            "Disable automatic std/prelude opening throughout each checked file "
            "(entry and library modules)."
        ),
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    # ``--dry-run`` is meaningless here: ``check`` is already side-effect free.
    del _dry_run
    if not file:
        _missing_arguments(["check"], ["FILE"])
    # Imported lazily: pulls in the AgL DSL (runtime, jsonschema), which would
    # otherwise slow every non-AgL ``agm`` invocation's startup.
    import agm.commands.check as check_command

    check_command.run(
        CheckArgs(
            files=file,
            module_paths=module_paths,
            no_stdlib=no_stdlib,
        )
    )


@worktree_app.command(name="rm")
def worktree_rm(
    branch: str | None = typer.Argument(
        None,
        metavar="BRANCH",
        autocompletion=completion.complete_close_branch,
    ),
    force: bool = typer.Option(
        False, "-f", "--force", help="Force removal of locked or dirty worktrees."
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    worktree_remove_command.run(
        WorktreeRemoveArgs(
            force=force,
            branch=_require_value(branch, command_path=["wt", "rm"], name="branch"),
        )
    )


@worktree_app.command(name="remove")
def worktree_remove(
    branch: str | None = typer.Argument(
        None,
        metavar="BRANCH",
        autocompletion=completion.complete_close_branch,
    ),
    force: bool = typer.Option(
        False, "-f", "--force", help="Force removal of locked or dirty worktrees."
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    worktree_remove_command.run(
        WorktreeRemoveArgs(
            force=force,
            branch=_require_value(branch, command_path=["worktree", "remove"], name="branch"),
        )
    )


@dep_app.callback(invoke_without_command=True)
def dep_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _group_help(ctx, "dep")


@dep_app.command(name="list")
def dep_list(
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show checkout paths."),
    list_all: bool = typer.Option(False, "--all", help="List all dependency checkouts on disk."),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    dep_list_command.run(verbose=verbose, all_checkouts=list_all)


@dep_app.command(name="new")
def new_dep(
    repo_url: str | None = typer.Argument(None, metavar="REPO_URL"),
    branch: str | None = typer.Option(
        None, "-b", "--branch", help="Clone BRANCH instead of the default branch."
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    dep_new_command.run(
        DepNewArgs(
            branch=branch,
            repo_url=_require_value(
                repo_url,
                command_path=["dep", "new"],
                name="repo-url",
            ),
        )
    )


@dep_app.command(name="switch")
def dep_switch(
    dep: str | None = typer.Argument(
        None,
        metavar="DEP",
        autocompletion=completion.complete_dep_name,
    ),
    branch: str | None = typer.Argument(
        None,
        metavar="BRANCH",
        autocompletion=completion.complete_dep_branch,
    ),
    create_branch: bool = typer.Option(
        False,
        "-b",
        "--branch",
        help="Create DEP's BRANCH from the dependency's default branch before adding it.",
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    if dep is None or branch is None:
        _missing_arguments(["dep", "switch"], ["dep", "branch"])
    dep_switch_command.run(DepSwitchArgs(dep=dep, branch=branch, create_branch=create_branch))


@dep_app.command(name="rm")
def dep_rm(
    target: str | None = typer.Argument(
        None,
        metavar="TARGET",
        autocompletion=completion.complete_dep_target,
    ),
    all: bool = typer.Option(False, "--all", help="Remove the entire dependency directory."),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _run_dep_remove(command_path=["dep", "rm"], target=target, all=all)


@dep_app.command(name="remove")
def dep_remove(
    target: str | None = typer.Argument(
        None,
        metavar="TARGET",
        autocompletion=completion.complete_dep_target,
    ),
    all: bool = typer.Option(False, "--all", help="Remove the entire dependency directory."),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _run_dep_remove(command_path=["dep", "remove"], target=target, all=all)


def _run_dep_remove(*, command_path: list[str], target: str | None, all: bool) -> None:
    dep_remove_command.run(
        DepRemoveArgs(
            all=all,
            target=_require_value(
                target,
                command_path=command_path,
                name="target",
            ),
        )
    )


@pkg_app.callback(invoke_without_command=True)
def pkg_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _group_help(ctx, "pkg")


@pkg_app.command(name="check")
def pkg_check(
    directory: str | None = typer.Argument(
        None,
        metavar="DIR",
        autocompletion=completion.complete_dir_argument,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    import agm.commands.pkg.check as pkg_check_command

    pkg_check_command.run(PkgCheckArgs(directory=directory))


@pkg_app.command(name="init")
def pkg_init(
    directory: str | None = typer.Argument(
        None,
        metavar="DIR",
        autocompletion=completion.complete_dir_argument,
    ),
    name: str | None = typer.Option(None, "--name", metavar="NAME"),
    version: str = typer.Option("0.1.0", "--version", metavar="VERSION"),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    import agm.commands.pkg.init as pkg_init_command

    pkg_init_command.run(PkgInitArgs(directory=directory, name=name, version=version))


@pkg_app.command(name="create")
def pkg_create(
    directory: str | None = typer.Argument(
        None,
        metavar="DIR",
        autocompletion=completion.complete_dir_argument,
    ),
    output: str | None = typer.Option(None, "-o", "--output", metavar="FILE"),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    import agm.commands.pkg.create as pkg_create_command

    pkg_create_command.run(PkgCreateArgs(directory=directory, output=output))


@pkg_app.command(name="install")
def pkg_install(
    source: str | None = typer.Argument(
        None, metavar="SRC", autocompletion=completion.complete_package_source
    ),
    editable: bool = typer.Option(False, "--editable", help="Activate a live package directory."),
    shadow: bool = typer.Option(False, "--shadow", help="Replace conflicting package commands."),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    import agm.commands.pkg.install as pkg_install_command

    pkg_install_command.run(
        PkgInstallArgs(
            source=_require_value(source, command_path=["pkg", "install"], name="source"),
            editable=editable,
            shadow=shadow,
        )
    )


@pkg_app.command(name="uninstall")
def pkg_uninstall(
    name: str | None = typer.Argument(None, metavar="NAME"),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    import agm.commands.pkg.uninstall as pkg_uninstall_command

    pkg_uninstall_command.run(
        PkgUninstallArgs(name=_require_value(name, command_path=["pkg", "uninstall"], name="name"))
    )


@pkg_app.command(name="list")
def pkg_list(
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    import agm.commands.pkg.list as pkg_list_command

    pkg_list_command.run(PkgListArgs())


@pkg_app.command(name="info")
def pkg_info(
    name: str | None = typer.Argument(None, metavar="NAME"),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    import agm.commands.pkg.info as pkg_info_command

    pkg_info_command.run(
        PkgInfoArgs(name=_require_value(name, command_path=["pkg", "info"], name="name"))
    )


@sync_app.callback(invoke_without_command=True)
def sync_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _group_help(ctx, "sync")


@sync_app.command(name="fetch")
def sync_fetch(_help: bool = _help_option(), _dry_run: bool = _dry_run_option()) -> None:
    del _help
    del _dry_run
    sync_fetch_command.run(object())


@sync_app.command(name="pull")
def sync_pull(_help: bool = _help_option(), _dry_run: bool = _dry_run_option()) -> None:
    del _help
    del _dry_run
    sync_pull_command.run(object())


@app.command()
def review(
    command_name: str | None = typer.Argument(None, metavar="COMMAND"),
    runner: str | None = typer.Option(None, "--runner", help="Review runner command."),
    scope: str | None = typer.Option(None, "--scope", help="Review scope."),
    aspects: str | None = typer.Option(None, "--aspects", help="Review aspects."),
    extra_aspects: str | None = typer.Option(
        None,
        "--extra-aspects",
        help="Additional review aspects appended to the defaults.",
    ),
    prompt: str | None = typer.Option(None, "--prompt", help="Inline review prompt."),
    prompt_file: str | None = typer.Option(
        None,
        "--prompt-file",
        help="Review prompt file.",
        autocompletion=completion.complete_path_argument,
    ),
    extra_prompt: str | None = typer.Option(
        None,
        "--extra-prompt",
        help="Extra inline review prompt content.",
    ),
    extra_prompt_file: str | None = typer.Option(
        None,
        "--extra-prompt-file",
        help="Extra review prompt file.",
        autocompletion=completion.complete_path_argument,
    ),
    review_file: str | None = typer.Option(
        None,
        "--review-file",
        help="Write review output to FILE, 'auto', or 'none'.",
        autocompletion=completion.complete_path_argument,
    ),
    no_review_file: bool = typer.Option(
        False,
        "--no-review-file",
        help="Disable saving review output.",
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _validate_prompt_options(
        command_path=["review"],
        prompt=prompt,
        prompt_file=prompt_file,
        extra_prompt=extra_prompt,
        extra_prompt_file=extra_prompt_file,
    )
    _reject_option_conflict(
        ["review"],
        "--no-review-file",
        "--review-file",
        conflicting=no_review_file and review_file is not None,
    )
    review_command.run(
        ReviewArgs(
            runner=runner,
            scope=scope,
            aspects=aspects,
            extra_aspects=extra_aspects,
            prompt=prompt,
            prompt_file=prompt_file,
            extra_prompt=extra_prompt,
            extra_prompt_file=extra_prompt_file,
            command_name=command_name,
            review_file=review_file,
            no_review_file=no_review_file,
        )
    )


@app.command()
def revise(
    command_name_or_review_file: str | None = typer.Argument(
        None,
        metavar="COMMAND_OR_REVIEW_FILE",
        autocompletion=completion.complete_revise_command_or_review_file,
    ),
    review_file: str | None = typer.Argument(
        None,
        metavar="REVIEW_FILE",
        autocompletion=completion.complete_path_argument,
    ),
    runner: str | None = typer.Option(None, "--runner", help="Revision runner command."),
    prompt: str | None = typer.Option(None, "--prompt", help="Inline revision prompt."),
    prompt_file: str | None = typer.Option(
        None,
        "--prompt-file",
        help="Revision prompt file.",
        autocompletion=completion.complete_path_argument,
    ),
    extra_prompt: str | None = typer.Option(
        None,
        "--extra-prompt",
        help="Extra inline revision prompt content.",
    ),
    extra_prompt_file: str | None = typer.Option(
        None,
        "--extra-prompt-file",
        help="Extra revision prompt file.",
        autocompletion=completion.complete_path_argument,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    command_name = command_name_or_review_file if review_file is not None else None
    resolved_review_file = review_file or command_name_or_review_file
    _validate_prompt_options(
        command_path=["revise"],
        prompt=prompt,
        prompt_file=prompt_file,
        extra_prompt=extra_prompt,
        extra_prompt_file=extra_prompt_file,
    )
    revise_command.run(
        ReviseArgs(
            review_file=_require_value(
                resolved_review_file,
                command_path=["revise"],
                name="review_file",
            ),
            runner=runner,
            prompt=prompt,
            prompt_file=prompt_file,
            extra_prompt=extra_prompt,
            extra_prompt_file=extra_prompt_file,
            command_name=command_name,
        )
    )


@app.command()
def refine(
    command_name: str | None = typer.Argument(None, metavar="COMMAND"),
    max_steps: str | None = typer.Option(
        None,
        "--max-steps",
        help="Maximum revision attempts (default: 12). Use 'unlimited' for no limit.",
    ),
    no_max_steps: bool = typer.Option(
        False,
        "--no-max-steps",
        help="Disable the step limit (run until COMPLETE).",
    ),
    runner: str | None = typer.Option(
        None,
        "--runner",
        help="Runner command for both review and revise.",
    ),
    reviewer: str | None = typer.Option(None, "--reviewer", help="Review runner command."),
    reviser: str | None = typer.Option(None, "--reviser", help="Revision runner command."),
    scope: str | None = typer.Option(None, "--scope", help="Review scope."),
    aspects: str | None = typer.Option(None, "--aspects", help="Review aspects."),
    review_prompt: str | None = typer.Option(
        None,
        "--review-prompt",
        help="Inline review prompt.",
    ),
    review_prompt_file: str | None = typer.Option(
        None,
        "--review-prompt-file",
        help="Review prompt file.",
        autocompletion=completion.complete_path_argument,
    ),
    extra_review_prompt: str | None = typer.Option(
        None,
        "--extra-review-prompt",
        help="Extra inline review prompt content.",
    ),
    extra_review_prompt_file: str | None = typer.Option(
        None,
        "--extra-review-prompt-file",
        help="Extra review prompt file.",
        autocompletion=completion.complete_path_argument,
    ),
    revise_prompt: str | None = typer.Option(
        None,
        "--revise-prompt",
        help="Inline revision prompt.",
    ),
    revise_prompt_file: str | None = typer.Option(
        None,
        "--revise-prompt-file",
        help="Revision prompt file.",
        autocompletion=completion.complete_path_argument,
    ),
    extra_revise_prompt: str | None = typer.Option(
        None,
        "--extra-revise-prompt",
        help="Extra inline revision prompt content.",
    ),
    extra_revise_prompt_file: str | None = typer.Option(
        None,
        "--extra-revise-prompt-file",
        help="Extra revision prompt file.",
        autocompletion=completion.complete_path_argument,
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Write command output to this log file.",
        autocompletion=completion.complete_path_argument,
    ),
    no_log: bool = typer.Option(False, "--no-log", help="Disable command output logging."),
    save_review: bool | None = typer.Option(
        None,
        "--save-review/--no-save-review",
        help="Save each review output to the default review file path.",
    ),
    review_file: str | None = typer.Option(
        None,
        "--review-file",
        help="Save each review output to this path, or use auto/none.",
        autocompletion=completion.complete_path_argument,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _validate_refine_prompt_options(
        command_path=["refine"],
        prompt_name="review",
        prompt=review_prompt,
        prompt_file=review_prompt_file,
        extra_prompt=extra_review_prompt,
        extra_prompt_file=extra_review_prompt_file,
    )
    _validate_refine_prompt_options(
        command_path=["refine"],
        prompt_name="revise",
        prompt=revise_prompt,
        prompt_file=revise_prompt_file,
        extra_prompt=extra_revise_prompt,
        extra_prompt_file=extra_revise_prompt_file,
    )
    _reject_option_conflict(
        ["refine"], "--no-log", "--log-file", conflicting=no_log and log_file is not None
    )
    _reject_option_conflict(
        ["refine"],
        "--no-max-steps",
        "--max-steps",
        conflicting=no_max_steps and max_steps is not None,
    )
    parsed_max_steps = _parse_max_steps(max_steps, command_path=["refine"], name="--max-steps")
    effective_no_max_steps = no_max_steps or (max_steps is not None and parsed_max_steps is None)
    refine_command.run(
        RefineArgs(
            max_steps=parsed_max_steps,
            no_max_steps=effective_no_max_steps,
            runner=runner,
            reviewer=reviewer,
            reviser=reviser,
            scope=scope,
            aspects=aspects,
            review_prompt=review_prompt,
            review_prompt_file=review_prompt_file,
            extra_review_prompt=extra_review_prompt,
            extra_review_prompt_file=extra_review_prompt_file,
            revise_prompt=revise_prompt,
            revise_prompt_file=revise_prompt_file,
            extra_revise_prompt=extra_revise_prompt,
            extra_revise_prompt_file=extra_revise_prompt_file,
            command_name=command_name,
            no_log=no_log,
            log_file=log_file,
            save_review=save_review,
            review_file=review_file,
        )
    )


@app.command(context_settings=_RUN_CONTEXT_SETTINGS)
def loop(
    ctx: typer.Context,
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    raw_args = list(ctx.args)
    if not raw_args:
        print_help_for_command_path(["loop"])
        raise typer.Exit()
    if raw_args[0] == "select":
        loop_select_command.run(_parse_loop_select_args(raw_args[1:]))
        return
    if raw_args[0] == "run":
        loop_run_command.run(
            _parse_loop_args(raw_args[1:], command_path=["loop", "run"], command_optional=True)
        )
        return
    if raw_args[0] == "step":
        loop_step_command.run(
            _parse_loop_args(raw_args[1:], command_path=["loop", "step"], command_optional=True)
        )
        return
    loop_command.run(_parse_loop_args(raw_args, command_path=["loop"]))


@app.command()
def init(
    arg1: str | None = typer.Argument(None, metavar="arg"),
    arg2: str | None = typer.Argument(None, metavar="arg"),
    embedded: bool = typer.Option(False, "--embedded", help="Force the embedded layout."),
    split: bool = typer.Option(False, "--split", help="Force the split layout."),
    clone: bool = typer.Option(
        False,
        "--clone",
        help="Initialize a new project directory derived from the repository URL.",
    ),
    branch: str | None = typer.Option(
        None, "-b", "--branch", help="Clone this branch when a repository URL is provided."
    ),
    no_config_git: bool = typer.Option(
        False,
        "--no-config-git",
        help="Do not create a git repository in the config/ directory.",
    ),
    no_notes_git: bool = typer.Option(
        False,
        "--no-notes-git",
        help="Do not create a git repository in the notes/ directory.",
    ),
    no_repo_git: bool = typer.Option(
        False,
        "--no-repo-git",
        help="Do not create a git repository in the repo/ directory.",
    ),
    no_git_init: bool = typer.Option(
        False,
        "--no-git-init",
        help="Do not create git repositories in repo/, config/, and notes/.",
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _reject_option_conflict(["init"], "--embedded", "--split", conflicting=embedded and split)
    positional: list[str] = [] if arg1 is None else [arg1] if arg2 is None else [arg1, arg2]
    init_command.run(
        InitArgs(
            positional=positional,
            branch=branch,
            embedded=embedded,
            split=split,
            clone=clone,
            no_repo_git=no_repo_git,
            no_config_git=no_config_git,
            no_notes_git=no_notes_git,
            no_git_init=no_git_init,
        )
    )


@app.command(context_settings=_RUN_CONTEXT_SETTINGS)
def run(
    run_command_args: list[str] | None = typer.Argument(
        None,
        metavar="CMD",
        autocompletion=completion.complete_run_command,
    ),
    no_sandbox: bool = typer.Option(
        False, "--no-sandbox", help="Run the command directly without srt sandboxing."
    ),
    no_patch: bool = typer.Option(False, "--no-patch", help="Skip filesystem allowWrite patching."),
    settings_file: Path | None = typer.Option(
        None,
        "-f",
        "--file",
        help="Use this settings file directly.",
        autocompletion=completion.complete_path_argument,
    ),
    memory: str | None = typer.Option(
        None,
        "--memory",
        help=(
            "Set MemoryMax inside delegated systemd-run; use 0 for a zero limit or "
            "unlimited for no memory cap."
        ),
    ),
    swap: str | None = typer.Option(
        None,
        "--swap",
        help=(
            "Set MemorySwapMax inside delegated systemd-run; default is 0 in sandbox mode, "
            "or use unlimited for no swap cap."
        ),
    ),
    no_memory_limit: bool = typer.Option(
        False,
        "--no-memory-limit",
        help="Do not set MemoryMax.",
    ),
    no_swap_limit: bool = typer.Option(
        False,
        "--no-swap-limit",
        help="Do not set MemorySwapMax.",
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    command = [] if run_command_args is None else list(run_command_args)
    if command and command[0].startswith("-") and command[0] != "--":
        exit_with_usage_error(["run"], f"error: unrecognized arguments: {' '.join(command)}")
    typed_args = RunArgs(
        run_command=command,
        no_sandbox=no_sandbox,
        no_patch=no_patch,
        memory=memory,
        swap=swap,
        no_memory_limit=no_memory_limit,
        no_swap_limit=no_swap_limit,
        settings_file=str(settings_file) if settings_file is not None else None,
    )
    if not run_command.normalize_run_command(list(typed_args.run_command)):
        print_help_for_command_path(["run"])
        raise typer.Exit()
    run_command.run(typed_args)


@tmux_app.callback(invoke_without_command=True)
def tmux_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    _group_help(ctx, "tmux")


@tmux_app.command(name="open")
def tmux_open(
    session_name: str | None = typer.Argument(
        None,
        metavar="SESSION",
        autocompletion=completion.complete_tmux_session,
    ),
    detach: bool = typer.Option(False, "-d", "--detach", help="Create the session detached."),
    pane_count: str | None = typer.Option(
        None,
        "-n",
        "--num-panes",
        help="Create the session with this many panes.",
        autocompletion=completion.complete_pane_count,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    tmux_open_command.run(
        TmuxOpenArgs(detach=detach, pane_count=pane_count, session_name=session_name)
    )


@tmux_app.command(name="close")
def tmux_close(
    session_name: str | None = typer.Argument(
        None,
        metavar="SESSION",
        autocompletion=completion.complete_tmux_session,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    tmux_close_command.run(
        TmuxCloseArgs(
            session_name=_require_value(
                session_name,
                command_path=["tmux", "close"],
                name="session",
            )
        )
    )


@tmux_app.command(name="layout")
def tmux_layout(
    pane_count: str | None = typer.Argument(
        None,
        metavar="PANES",
        autocompletion=completion.complete_pane_count,
    ),
    window_id: str | None = typer.Option(
        None,
        "-w",
        "--window",
        help="Target a specific tmux window id.",
        autocompletion=completion.complete_tmux_window,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    tmux_layout_command.run(
        TmuxLayoutArgs(
            pane_count=_require_value(pane_count, command_path=["tmux", "layout"], name="panes"),
            window_id=window_id,
        )
    )


app.add_typer(config_app, name="config")
app.add_typer(workspace_app, name="workspace")
app.add_typer(workspace_app, name="wsp")
app.add_typer(worktree_app, name="wt")
app.add_typer(worktree_app, name="worktree")
app.add_typer(sync_app, name="sync")
app.add_typer(dep_app, name="dep")
app.add_typer(pkg_app, name="pkg")
app.add_typer(tmux_app, name="tmux")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
