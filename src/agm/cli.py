"""AGM command-line interface implemented with Typer."""

from __future__ import annotations

from collections.abc import Sequence

import typer

import agm.commands.close as close_command
import agm.commands.config.copy as config_copy_command
import agm.commands.dep.new as dep_new_command
import agm.commands.dep.remove as dep_remove_command
import agm.commands.dep.switch as dep_switch_command
import agm.commands.fetch as fetch_command
import agm.commands.init as init_command
import agm.commands.open as open_command
import agm.commands.run as run_command
import agm.commands.tmux.close as tmux_close_command
import agm.commands.tmux.layout as tmux_layout_command
import agm.commands.tmux.open as tmux_open_command
import agm.commands.worktree.new as worktree_new_command
import agm.commands.worktree.remove as worktree_remove_command
import agm.commands.worktree.setup as worktree_setup_command
from agm import parser as parser_helpers
from agm.commands.args import (
    CloseArgs,
    ConfigCopyArgs,
    DepNewArgs,
    DepRemoveArgs,
    DepSwitchArgs,
    InitArgs,
    OpenArgs,
    RunArgs,
    TmuxCloseArgs,
    TmuxLayoutArgs,
    TmuxOpenArgs,
    WorktreeNewArgs,
    WorktreeRemoveArgs,
    WorktreeSetupArgs,
)
from agm.parser import (
    exit_with_usage_error,
    print_command_help,
    print_help_for_command_path,
    print_overview,
)

_HELP_TEXTS = parser_helpers._HELP_TEXTS
_HELP_ALIASES = parser_helpers._HELP_ALIASES
_COMMAND_OVERVIEW = parser_helpers._COMMAND_OVERVIEW

_BASE_CONTEXT_SETTINGS: dict[str, bool | list[str]] = {"help_option_names": []}
_EXTRA_ARGS_CONTEXT_SETTINGS: dict[str, bool | list[str]] = {
    "help_option_names": [],
    "allow_extra_args": True,
    "ignore_unknown_options": False,
}
_RUN_CONTEXT_SETTINGS: dict[str, bool | list[str]] = {
    "help_option_names": [],
    "allow_extra_args": True,
    "ignore_unknown_options": True,
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


def _print_context_help(
    ctx: typer.Context,
    param: object,
    value: bool,
) -> None:
    del param
    if not value or ctx.resilient_parsing:
        return
    command_path = _command_path_from_context(ctx)
    if command_path:
        print_help_for_command_path(command_path)
    else:
        print_overview()
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


def _missing_arguments(command_path: Sequence[str], names: Sequence[str]) -> None:
    joined = ", ".join(names)
    exit_with_usage_error(command_path, f"error: the following arguments are required: {joined}")


def _unexpected_arguments(command_path: Sequence[str], args: Sequence[str]) -> None:
    exit_with_usage_error(command_path, f"error: unrecognized arguments: {' '.join(args)}")


def _single_required_arg(
    ctx: typer.Context,
    *,
    command_path: Sequence[str],
    name: str,
) -> str:
    extra_args = list(ctx.args)
    if not extra_args:
        _missing_arguments(command_path, [name])
    if len(extra_args) > 1:
        _unexpected_arguments(command_path, extra_args[1:])
    return extra_args[0]


def _one_or_two_args(
    ctx: typer.Context,
    *,
    command_path: Sequence[str],
    name: str,
) -> list[str]:
    extra_args = list(ctx.args)
    if not extra_args:
        _missing_arguments(command_path, [name])
    if len(extra_args) > 2:
        _unexpected_arguments(command_path, extra_args[2:])
    return extra_args


app = typer.Typer(
    add_completion=False,
    context_settings=_BASE_CONTEXT_SETTINGS,
    invoke_without_command=True,
)

config_app = typer.Typer(
    add_completion=False,
    context_settings=_BASE_CONTEXT_SETTINGS,
    invoke_without_command=True,
)
worktree_app = typer.Typer(
    add_completion=False,
    context_settings=_BASE_CONTEXT_SETTINGS,
    invoke_without_command=True,
)
dep_app = typer.Typer(
    add_completion=False,
    context_settings=_BASE_CONTEXT_SETTINGS,
    invoke_without_command=True,
)
tmux_app = typer.Typer(
    add_completion=False,
    context_settings=_BASE_CONTEXT_SETTINGS,
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
) -> None:
    del _help
    if ctx.invoked_subcommand is None:
        print_overview()
        raise typer.Exit()


@app.command(context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def help(
    ctx: typer.Context,
    _help: bool = _help_option(),
) -> None:
    del _help
    help_command = list(ctx.args)
    if not help_command:
        print_overview()
        raise typer.Exit()
    try:
        print_help_for_command_path(help_command)
    except ValueError:
        print_command_help(" ".join(help_command))
    raise typer.Exit()


@app.command(context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def open(
    ctx: typer.Context,
    detached: bool = typer.Option(
        False, "-d", "--detach", "--detached", help="Open the session detached."
    ),
    pane_count: str | None = typer.Option(
        None, "-n", "--num-panes", help="Create the session with this many panes."
    ),
    parent: str | None = typer.Option(
        None, "-p", "--parent", help="Base a new branch on this checkout."
    ),
    _help: bool = _help_option(),
) -> None:
    del _help
    branch = _single_required_arg(ctx, command_path=["open"], name="target")
    open_command.run(
        OpenArgs(
            detached=detached,
            pane_count=pane_count,
            parent=parent,
            branch=branch,
        )
    )


@app.command(context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def close(
    ctx: typer.Context,
    _help: bool = _help_option(),
) -> None:
    del _help
    branch = _single_required_arg(ctx, command_path=["close"], name="branch")
    close_command.run(CloseArgs(branch=branch))


@config_app.callback(invoke_without_command=True)
def config_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
) -> None:
    del _help
    if ctx.invoked_subcommand is None:
        print_help_for_command_path(["config"])
        raise typer.Exit()


@config_app.command(name="cp", context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def config_cp(
    ctx: typer.Context,
    project_dir: str | None = typer.Option(
        None, "-d", "--dir", help="Read config from this project."
    ),
    _help: bool = _help_option(),
) -> None:
    del _help
    dirname = _single_required_arg(ctx, command_path=["config", "cp"], name="dirname")
    config_copy_command.run(
        ConfigCopyArgs(config_command="cp", project_dir=project_dir, dirname=dirname)
    )


@config_app.command(name="copy", context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def config_copy(
    ctx: typer.Context,
    project_dir: str | None = typer.Option(
        None, "-d", "--dir", help="Read config from this project."
    ),
    _help: bool = _help_option(),
) -> None:
    del _help
    dirname = _single_required_arg(ctx, command_path=["config", "copy"], name="dirname")
    config_copy_command.run(
        ConfigCopyArgs(config_command="copy", project_dir=project_dir, dirname=dirname)
    )


@worktree_app.callback(invoke_without_command=True)
def worktree_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
) -> None:
    del _help
    if ctx.invoked_subcommand is None:
        print_help_for_command_path([ctx.info_name or "worktree"])
        raise typer.Exit()


@worktree_app.command(context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def new(
    ctx: typer.Context,
    worktrees_dir: str | None = typer.Option(
        None, "-d", "--dir", help="Create the worktree under DIR."
    ),
    _help: bool = _help_option(),
) -> None:
    del _help
    command_path = _command_path_from_context(ctx)
    branch = _single_required_arg(ctx, command_path=command_path, name="branch")
    worktree_new_command.run(WorktreeNewArgs(worktrees_dir=worktrees_dir, branch=branch))


@worktree_app.command(context_settings=_BASE_CONTEXT_SETTINGS)
def setup(
    _help: bool = _help_option(),
) -> None:
    del _help
    worktree_setup_command.run(WorktreeSetupArgs(wt_command="setup"))


@worktree_app.command(name="rm", context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def worktree_rm(
    ctx: typer.Context,
    force: bool = typer.Option(
        False, "-f", "--force", help="Force removal of locked or dirty worktrees."
    ),
    _help: bool = _help_option(),
) -> None:
    del _help
    command_path = _command_path_from_context(ctx)
    branch = _single_required_arg(ctx, command_path=command_path, name="branch")
    worktree_remove_command.run(WorktreeRemoveArgs(force=force, branch=branch))


@worktree_app.command(name="remove", context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def worktree_remove(
    ctx: typer.Context,
    force: bool = typer.Option(
        False, "-f", "--force", help="Force removal of locked or dirty worktrees."
    ),
    _help: bool = _help_option(),
) -> None:
    del _help
    command_path = _command_path_from_context(ctx)
    branch = _single_required_arg(ctx, command_path=command_path, name="branch")
    worktree_remove_command.run(WorktreeRemoveArgs(force=force, branch=branch))


@dep_app.callback(invoke_without_command=True)
def dep_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
) -> None:
    del _help
    if ctx.invoked_subcommand is None:
        print_help_for_command_path(["dep"])
        raise typer.Exit()


@dep_app.command(name="new", context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def new_dep(
    ctx: typer.Context,
    branch: str | None = typer.Option(
        None, "-b", "--branch", help="Clone BRANCH instead of the default branch."
    ),
    _help: bool = _help_option(),
) -> None:
    del _help
    repo_url = _single_required_arg(ctx, command_path=["dep", "new"], name="repo-url")
    dep_new_command.run(DepNewArgs(branch=branch, repo_url=repo_url))


@dep_app.command(name="switch", context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def dep_switch(
    ctx: typer.Context,
    create_branch: bool = typer.Option(
        False, "-b", "--branch", help="Create the branch from the default branch."
    ),
    _help: bool = _help_option(),
) -> None:
    del _help
    extra_args = list(ctx.args)
    if len(extra_args) < 2:
        _missing_arguments(["dep", "switch"], ["dep", "branch"])
    if len(extra_args) > 2:
        _unexpected_arguments(["dep", "switch"], extra_args[2:])
    dep_switch_command.run(
        DepSwitchArgs(dep=extra_args[0], branch=extra_args[1], create_branch=create_branch)
    )


@dep_app.command(name="rm", context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def dep_rm(
    ctx: typer.Context,
    all: bool = typer.Option(False, "--all", help="Remove the entire dependency directory."),
    _help: bool = _help_option(),
) -> None:
    del _help
    target = _single_required_arg(ctx, command_path=["dep", "rm"], name="target")
    dep_remove_command.run(DepRemoveArgs(all=all, target=target))


@app.command(context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def fetch(
    ctx: typer.Context,
    _help: bool = _help_option(),
) -> None:
    del _help
    if ctx.args:
        _unexpected_arguments(["fetch"], list(ctx.args))
    fetch_command.run(object())


@app.command(context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def init(
    ctx: typer.Context,
    embedded: bool = typer.Option(False, "--embedded", help="Force the embedded layout."),
    workspace: bool = typer.Option(False, "--workspace", help="Force the workspace layout."),
    branch: str | None = typer.Option(
        None, "-b", "--branch", help="Clone this branch when a repository URL is provided."
    ),
    _help: bool = _help_option(),
) -> None:
    del _help
    if embedded and workspace:
        exit_with_usage_error(["init"], "error: --embedded and --workspace are mutually exclusive")
    positional = _one_or_two_args(ctx, command_path=["init"], name="arg")
    init_command.run(
        InitArgs(
            positional=positional,
            branch=branch,
            embedded=embedded,
            workspace=workspace,
        )
    )


@app.command(context_settings=_RUN_CONTEXT_SETTINGS)
def run(
    ctx: typer.Context,
    no_patch: bool = typer.Option(
        False, "--no-patch", help="Skip filesystem allowWrite patching."
    ),
    settings_file: str | None = typer.Option(
        None, "-f", "--file", help="Use this settings file directly."
    ),
    _help: bool = _help_option(),
) -> None:
    del _help
    extra_args = list(ctx.args)
    if extra_args and extra_args[0].startswith("-") and extra_args[0] != "--":
        _unexpected_arguments(["run"], extra_args)
    run_args = RunArgs(run_command=extra_args, no_patch=no_patch, settings_file=settings_file)
    if not run_command.normalize_run_command(list(run_args.run_command)):
        print_help_for_command_path(["run"])
        raise typer.Exit()
    run_command.run(run_args)


@tmux_app.callback(invoke_without_command=True)
def tmux_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
) -> None:
    del _help
    if ctx.invoked_subcommand is None:
        print_help_for_command_path(["tmux"])
        raise typer.Exit()


@tmux_app.command(name="open", context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def tmux_open(
    ctx: typer.Context,
    detach: bool = typer.Option(False, "-d", "--detach", help="Create the session detached."),
    pane_count: str | None = typer.Option(
        None, "-n", "--num-panes", help="Create the session with this many panes."
    ),
    _help: bool = _help_option(),
) -> None:
    del _help
    extra_args = list(ctx.args)
    if len(extra_args) > 1:
        _unexpected_arguments(["tmux", "open"], extra_args[1:])
    session_name = extra_args[0] if extra_args else None
    tmux_open_command.run(
        TmuxOpenArgs(detach=detach, pane_count=pane_count, session_name=session_name)
    )


@tmux_app.command(name="close", context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def tmux_close(
    ctx: typer.Context,
    _help: bool = _help_option(),
) -> None:
    del _help
    session_name = _single_required_arg(ctx, command_path=["tmux", "close"], name="session")
    tmux_close_command.run(TmuxCloseArgs(session_name=session_name))


@tmux_app.command(name="layout", context_settings=_EXTRA_ARGS_CONTEXT_SETTINGS)
def tmux_layout(
    ctx: typer.Context,
    window_id: str | None = typer.Option(
        None, "-w", "--window", help="Target a specific tmux window id."
    ),
    _help: bool = _help_option(),
) -> None:
    del _help
    pane_count = _single_required_arg(ctx, command_path=["tmux", "layout"], name="panes")
    tmux_layout_command.run(TmuxLayoutArgs(pane_count=pane_count, window_id=window_id))


app.add_typer(config_app, name="config")
app.add_typer(worktree_app, name="wt")
app.add_typer(worktree_app, name="worktree")
app.add_typer(dep_app, name="dep")
app.add_typer(tmux_app, name="tmux")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
