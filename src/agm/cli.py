"""AGM command-line interface implemented with Typer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import typer

import agm.commands.close as close_command
import agm.commands.config.copy as config_copy_command
import agm.commands.dep.new as dep_new_command
import agm.commands.dep.remove as dep_remove_command
import agm.commands.dep.switch as dep_switch_command
import agm.commands.fetch as fetch_command
import agm.commands.init as init_command
import agm.commands.loop.next as loop_next_command
import agm.commands.loop.run as loop_command
import agm.commands.loop.run as loop_run_command
import agm.commands.loop.step as loop_step_command
import agm.commands.open as open_command
import agm.commands.run as run_command
import agm.commands.tmux.close as tmux_close_command
import agm.commands.tmux.layout as tmux_layout_command
import agm.commands.tmux.open as tmux_open_command
import agm.commands.worktree.new as worktree_new_command
import agm.commands.worktree.remove as worktree_remove_command
import agm.commands.worktree.setup as worktree_setup_command
from agm import completion
from agm import parser as parser_helpers
from agm.commands.args import (
    CloseArgs,
    ConfigCopyArgs,
    DepNewArgs,
    DepRemoveArgs,
    DepSwitchArgs,
    InitArgs,
    LoopArgs,
    LoopProgressArgs,
    OpenArgs,
    RunArgs,
    TmuxCloseArgs,
    TmuxLayoutArgs,
    TmuxOpenArgs,
    WorktreeNewArgs,
    WorktreeRemoveArgs,
    WorktreeSetupArgs,
)
from agm.core import dry_run
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


def _root_context(ctx: typer.Context) -> typer.Context:
    current = ctx
    while current.parent is not None:
        current = current.parent
    return current


def _print_context_help(ctx: typer.Context, param: object, value: bool) -> None:
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


def _set_dry_run(ctx: typer.Context, param: object, value: bool) -> None:
    del param
    root = _root_context(ctx)
    root_meta = cast(dict[str, bool], getattr(root, "meta"))
    enabled = value or bool(root_meta.get("dry_run"))
    root_meta["dry_run"] = enabled
    dry_run.set_enabled(enabled)


def _dry_run_option() -> bool:
    return typer.Option(
        False,
        "--dry-run",
        callback=_set_dry_run,
        expose_value=False,
        is_eager=True,
        help="Print commands and AGM operations without executing them.",
    )


def _missing_arguments(command_path: Sequence[str], names: Sequence[str]) -> None:
    joined = ", ".join(names)
    exit_with_usage_error(command_path, f"error: the following arguments are required: {joined}")


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


def _parse_loop_args(
    raw_args: list[str],
    *,
    command_path: Sequence[str],
    command_optional: bool = False,
) -> LoopArgs:
    runner: str | None = None
    selector: str | None = None
    tasks_dir: str | None = None
    no_log = False
    log_file: str | None = None
    index = 0

    while index < len(raw_args):
        token = raw_args[index]
        if token == "--":
            break
        if token == "--runner":
            runner, index = _loop_option_value(
                raw_args,
                index,
                command_path=command_path,
                option=token,
            )
            continue
        if token == "--selector":
            selector, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--tasks-dir":
            tasks_dir, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--log-file":
            log_file, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--no-log":
            no_log = True
            index += 1
            continue
        break

    remaining = raw_args[index:]
    if not remaining:
        if command_optional:
            if no_log and log_file is not None:
                exit_with_usage_error(
                    command_path, "error: --no-log and --log-file are mutually exclusive"
                )
            return LoopArgs(
                command_name=None,
                runner=runner,
                runner_args=[],
                selector=selector,
                tasks_dir=tasks_dir,
                no_log=no_log,
                log_file=log_file,
            )
        print_help_for_command_path(command_path)
        raise typer.Exit()
    command_name = remaining[0]
    runner_args = run_command.normalize_run_command(remaining[1:])
    if no_log and log_file is not None:
        exit_with_usage_error(
            command_path, "error: --no-log and --log-file are mutually exclusive"
        )
    return LoopArgs(
        command_name=command_name,
        runner=runner,
        runner_args=runner_args,
        selector=selector,
        tasks_dir=tasks_dir,
        no_log=no_log,
        log_file=log_file,
    )


def _parse_loop_next_args(
    raw_args: list[str], *, command_path: Sequence[str] = ("loop", "next")
) -> LoopProgressArgs:
    runner: str | None = None
    selector: str | None = None
    tasks_dir: str | None = None
    index = 0

    while index < len(raw_args):
        token = raw_args[index]
        if token == "--":
            break
        if token == "--runner":
            runner, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--selector":
            selector, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--tasks-dir":
            tasks_dir, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        break

    remaining = raw_args[index:]
    command_name: str | None = None
    runner_args: list[str] = []
    if remaining:
        command_name = remaining[0]
        runner_args = run_command.normalize_run_command(remaining[1:])
    return LoopProgressArgs(
        command_name=command_name,
        runner=runner,
        runner_args=runner_args,
        selector=selector,
        tasks_dir=tasks_dir,
    )


app = typer.Typer(context_settings=_BASE_CONTEXT_SETTINGS, invoke_without_command=True)

config_app = typer.Typer(context_settings=_BASE_CONTEXT_SETTINGS, invoke_without_command=True)
worktree_app = typer.Typer(context_settings=_BASE_CONTEXT_SETTINGS, invoke_without_command=True)
dep_app = typer.Typer(context_settings=_BASE_CONTEXT_SETTINGS, invoke_without_command=True)
tmux_app = typer.Typer(context_settings=_BASE_CONTEXT_SETTINGS, invoke_without_command=True)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    if ctx.invoked_subcommand is None:
        print_overview()
        raise typer.Exit()


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
        print_command_help(" ".join(help_command))
    raise typer.Exit()


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
        help="Base a new branch on this checkout.",
        autocompletion=completion.complete_worktree_branch,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    open_command.run(
        OpenArgs(
            detached=detached,
            pane_count=pane_count,
            parent=parent,
            branch=_require_value(target, command_path=["open"], name="target"),
        )
    )


@app.command()
def close(
    branch: str | None = typer.Argument(
        None,
        metavar="BRANCH",
        autocompletion=completion.complete_close_branch,
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    close_command.run(
        CloseArgs(
            branch=_require_value(
                branch,
                command_path=["close"],
                name="branch",
            )
        )
    )


@config_app.callback(invoke_without_command=True)
def config_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    if ctx.invoked_subcommand is None:
        print_help_for_command_path(["config"])
        raise typer.Exit()


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


@worktree_app.callback(invoke_without_command=True)
def worktree_callback(
    ctx: typer.Context,
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    if ctx.invoked_subcommand is None:
        print_help_for_command_path([ctx.info_name or "worktree"])
        raise typer.Exit()


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


@worktree_app.command()
def setup(_help: bool = _help_option(), _dry_run: bool = _dry_run_option()) -> None:
    del _help
    del _dry_run
    worktree_setup_command.run(WorktreeSetupArgs(wt_command="setup"))


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
    if ctx.invoked_subcommand is None:
        print_help_for_command_path(["dep"])
        raise typer.Exit()


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
        False, "-b", "--branch", help="Create the branch from the default branch."
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    if dep is None or branch is None:
        _missing_arguments(["dep", "switch"], ["dep", "branch"])
    assert dep is not None
    assert branch is not None
    dep_switch_command.run(
        DepSwitchArgs(dep=dep, branch=branch, create_branch=create_branch)
    )


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
    dep_remove_command.run(
        DepRemoveArgs(
            all=all,
            target=_require_value(
                target,
                command_path=["dep", "rm"],
                name="target",
            ),
        )
    )


@app.command()
def fetch(_help: bool = _help_option(), _dry_run: bool = _dry_run_option()) -> None:
    del _help
    del _dry_run
    fetch_command.run(object())


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
    if raw_args[0] == "next":
        loop_next_command.run(_parse_loop_next_args(raw_args[1:]))
        return
    if raw_args[0] == "run":
        loop_run_command.run(
            _parse_loop_args(
                raw_args[1:], command_path=["loop", "run"], command_optional=True
            )
        )
        return
    if raw_args[0] == "step":
        loop_step_command.run(_parse_loop_args(raw_args[1:], command_path=["loop", "step"]))
        return
    loop_command.run(_parse_loop_args(raw_args, command_path=["loop"]))


@app.command()
def init(
    arg1: str | None = typer.Argument(None, metavar="arg"),
    arg2: str | None = typer.Argument(None, metavar="arg"),
    embedded: bool = typer.Option(False, "--embedded", help="Force the embedded layout."),
    workspace: bool = typer.Option(False, "--workspace", help="Force the workspace layout."),
    branch: str | None = typer.Option(
        None, "-b", "--branch", help="Clone this branch when a repository URL is provided."
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    if embedded and workspace:
        exit_with_usage_error(["init"], "error: --embedded and --workspace are mutually exclusive")
    if arg1 is None:
        _missing_arguments(["init"], ["arg"])
    assert arg1 is not None
    positional: list[str] = [arg1] if arg2 is None else [arg1, arg2]
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
    run_command_args: list[str] | None = typer.Argument(
        None,
        metavar="CMD",
        autocompletion=completion.complete_run_command,
    ),
    no_sandbox: bool = typer.Option(
        False, "--no-sandbox", help="Run the command directly without srt sandboxing."
    ),
    no_patch: bool = typer.Option(
        False, "--no-patch", help="Skip filesystem allowWrite patching."
    ),
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
        help="Set MemoryMax; <= 0 disables memory limiting.",
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
    if ctx.invoked_subcommand is None:
        print_help_for_command_path(["tmux"])
        raise typer.Exit()


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
app.add_typer(worktree_app, name="wt")
app.add_typer(worktree_app, name="worktree")
app.add_typer(dep_app, name="dep")
app.add_typer(tmux_app, name="tmux")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
