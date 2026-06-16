"""AGM command-line interface implemented with Typer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import typer

import agm.commands.close as close_command
import agm.commands.config.copy as config_copy_command
import agm.commands.config.env as config_env_command
import agm.commands.config.update as config_update_command
import agm.commands.dep.list as dep_list_command
import agm.commands.dep.new as dep_new_command
import agm.commands.dep.remove as dep_remove_command
import agm.commands.dep.switch as dep_switch_command
import agm.commands.exec as exec_command
import agm.commands.fetch as fetch_command
import agm.commands.init as init_command
import agm.commands.list as list_command
import agm.commands.loop.run as loop_command
import agm.commands.loop.run as loop_run_command
import agm.commands.loop.select as loop_select_command
import agm.commands.loop.step as loop_step_command
import agm.commands.open as open_command
import agm.commands.pull as pull_command
import agm.commands.refine as refine_command
import agm.commands.repl as repl_command
import agm.commands.review as review_command
import agm.commands.revise as revise_command
import agm.commands.run as run_command
import agm.commands.setup as setup_command
import agm.commands.tmux.close as tmux_close_command
import agm.commands.tmux.layout as tmux_layout_command
import agm.commands.tmux.open as tmux_open_command
import agm.commands.worktree.new as worktree_new_command
import agm.commands.worktree.remove as worktree_remove_command
from agm import completion
from agm import parser as parser_helpers
from agm.commands.args import (
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
    LoopSelectArgs,
    OpenArgs,
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
from agm.config.general import parse_timeout
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
    no_selector = False
    tasks_dir: str | None = None
    no_log = False
    log_file: str | None = None
    prompt: str | None = None
    prompt_file: str | None = None
    selector_prompt: str | None = None
    selector_prompt_file: str | None = None
    extra_prompt: str | None = None
    extra_prompt_file: str | None = None
    extra_selector_prompt: str | None = None
    extra_selector_prompt_file: str | None = None
    timeout: float | None = None
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
        if token == "--no-selector":
            no_selector = True
            index += 1
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
        if token == "--prompt":
            prompt, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--prompt-file":
            prompt_file, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--selector-prompt":
            selector_prompt, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--selector-prompt-file":
            selector_prompt_file, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--extra-prompt":
            extra_prompt, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--extra-prompt-file":
            extra_prompt_file, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--extra-selector-prompt":
            extra_selector_prompt, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--extra-selector-prompt-file":
            extra_selector_prompt_file, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--timeout":
            timeout_str, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            try:
                timeout = parse_timeout(timeout_str)
            except ValueError as exc:
                exit_with_usage_error(command_path, f"error: {exc}")
            continue
        break

    if selector is not None and no_selector:
        exit_with_usage_error(
            command_path, "error: --selector and --no-selector are mutually exclusive"
        )
    if prompt is not None and prompt_file is not None:
        exit_with_usage_error(
            command_path, "error: --prompt and --prompt-file are mutually exclusive"
        )
    if selector_prompt is not None and selector_prompt_file is not None:
        exit_with_usage_error(
            command_path,
            "error: --selector-prompt and --selector-prompt-file are mutually exclusive",
        )
    if extra_prompt is not None and extra_prompt_file is not None:
        exit_with_usage_error(
            command_path,
            "error: --extra-prompt and --extra-prompt-file are mutually exclusive",
        )
    if extra_selector_prompt is not None and extra_selector_prompt_file is not None:
        exit_with_usage_error(
            command_path,
            "error: --extra-selector-prompt and "
            "--extra-selector-prompt-file are mutually exclusive",
        )
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
                no_selector=no_selector,
                tasks_dir=tasks_dir,
                no_log=no_log,
                log_file=log_file,
                prompt=prompt,
                prompt_file=prompt_file,
                selector_prompt=selector_prompt,
                selector_prompt_file=selector_prompt_file,
                extra_prompt=extra_prompt,
                extra_prompt_file=extra_prompt_file,
                extra_selector_prompt=extra_selector_prompt,
                extra_selector_prompt_file=extra_selector_prompt_file,
                timeout=timeout,
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
        no_selector=no_selector,
        tasks_dir=tasks_dir,
        no_log=no_log,
        log_file=log_file,
        prompt=prompt,
        prompt_file=prompt_file,
        selector_prompt=selector_prompt,
        selector_prompt_file=selector_prompt_file,
        extra_prompt=extra_prompt,
        extra_prompt_file=extra_prompt_file,
        extra_selector_prompt=extra_selector_prompt,
        extra_selector_prompt_file=extra_selector_prompt_file,
        timeout=timeout,
    )


def _parse_loop_select_args(
    raw_args: list[str], *, command_path: Sequence[str] = ("loop", "select")
) -> LoopSelectArgs:
    runner: str | None = None
    selector: str | None = None
    no_selector = False
    tasks_dir: str | None = None
    prompt: str | None = None
    prompt_file: str | None = None
    selector_prompt: str | None = None
    selector_prompt_file: str | None = None
    extra_prompt: str | None = None
    extra_prompt_file: str | None = None
    extra_selector_prompt: str | None = None
    extra_selector_prompt_file: str | None = None
    timeout: float | None = None
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
        if token == "--no-selector":
            no_selector = True
            index += 1
            continue
        if token == "--tasks-dir":
            tasks_dir, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--prompt":
            prompt, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--prompt-file":
            prompt_file, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--selector-prompt":
            selector_prompt, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--selector-prompt-file":
            selector_prompt_file, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--extra-prompt":
            extra_prompt, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--extra-prompt-file":
            extra_prompt_file, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--extra-selector-prompt":
            extra_selector_prompt, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--extra-selector-prompt-file":
            extra_selector_prompt_file, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            continue
        if token == "--timeout":
            timeout_str, index = _loop_option_value(
                raw_args, index, command_path=command_path, option=token
            )
            try:
                timeout = parse_timeout(timeout_str)
            except ValueError as exc:
                exit_with_usage_error(command_path, f"error: {exc}")
            continue
        break

    if selector is not None and no_selector:
        exit_with_usage_error(
            command_path, "error: --selector and --no-selector are mutually exclusive"
        )
    if prompt is not None and prompt_file is not None:
        exit_with_usage_error(
            command_path, "error: --prompt and --prompt-file are mutually exclusive"
        )
    if selector_prompt is not None and selector_prompt_file is not None:
        exit_with_usage_error(
            command_path,
            "error: --selector-prompt and --selector-prompt-file are mutually exclusive",
        )
    if extra_prompt is not None and extra_prompt_file is not None:
        exit_with_usage_error(
            command_path,
            "error: --extra-prompt and --extra-prompt-file are mutually exclusive",
        )
    if extra_selector_prompt is not None and extra_selector_prompt_file is not None:
        exit_with_usage_error(
            command_path,
            "error: --extra-selector-prompt and "
            "--extra-selector-prompt-file are mutually exclusive",
        )
    remaining = raw_args[index:]
    command_name: str | None = None
    runner_args: list[str] = []
    if remaining:
        command_name = remaining[0]
        runner_args = run_command.normalize_run_command(remaining[1:])
    return LoopSelectArgs(
        command_name=command_name,
        runner=runner,
        runner_args=runner_args,
        selector=selector,
        no_selector=no_selector,
        tasks_dir=tasks_dir,
        prompt=prompt,
        prompt_file=prompt_file,
        selector_prompt=selector_prompt,
        selector_prompt_file=selector_prompt_file,
        extra_prompt=extra_prompt,
        extra_prompt_file=extra_prompt_file,
        extra_selector_prompt=extra_selector_prompt,
        extra_selector_prompt_file=extra_selector_prompt_file,
        timeout=timeout,
    )


def _validate_prompt_options(
    *,
    command_path: Sequence[str],
    prompt: str | None,
    prompt_file: str | None,
    extra_prompt: str | None,
    extra_prompt_file: str | None,
) -> None:
    if prompt is not None and prompt_file is not None:
        exit_with_usage_error(
            command_path, "error: --prompt and --prompt-file are mutually exclusive"
        )
    if extra_prompt is not None and extra_prompt_file is not None:
        exit_with_usage_error(
            command_path,
            "error: --extra-prompt and --extra-prompt-file are mutually exclusive",
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
    if prompt is not None and prompt_file is not None:
        exit_with_usage_error(
            command_path,
            f"error: --{prompt_name}-prompt and --{prompt_name}-prompt-file "
            "are mutually exclusive",
        )
    if extra_prompt is not None and extra_prompt_file is not None:
        exit_with_usage_error(
            command_path,
            f"error: --extra-{prompt_name}-prompt and "
            f"--extra-{prompt_name}-prompt-file are mutually exclusive",
        )


def _parse_max_steps(
    value: str | None, *, command_path: Sequence[str], name: str
) -> int | None:
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
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help=(
            "Force remove the worktree (even with untracked files) and force delete the"
            " branch (git branch -D)."
        ),
    ),
    force_delete: bool = typer.Option(
        False,
        "-D",
        help="Force delete the branch (git branch -D) instead of safe delete (git branch -d).",
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
            ),
            force=force,
            force_delete=force_delete,
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


@app.command()
def setup(_help: bool = _help_option(), _dry_run: bool = _dry_run_option()) -> None:
    del _help
    del _dry_run
    setup_command.run()


def _exec_print_help(*, file: str | None, command: str | None) -> None:
    """Print exec help, optionally with program param section, then exit 0.

    When FILE or -c is provided and the source can be prepared + typechecked,
    appends the discovered ``Program parameters:`` section.  Degrades silently
    on any error (syntax errors, unreadable files, etc.).
    """
    from agm.agl import WorkflowRuntime
    from agm.commands.param_options import render_param_help_section
    from agm.core.fs import read_text_arg

    print_help_for_command_path(["exec"])

    source: str | None = None
    if command is not None:
        source = command
    elif file is not None:
        try:
            source = read_text_arg(Path(file))
        except SystemExit:
            source = None

    if source is not None:
        try:
            prepared = WorkflowRuntime.prepare(source)
            runtime = WorkflowRuntime()
            discovery = runtime.discover_params(prepared)
            if discovery.params:
                section = render_param_help_section(discovery.params)
                print(section, end="")
        except (Exception, SystemExit):
            pass

    raise SystemExit(0)


@app.command(name="exec", context_settings=_RUN_CONTEXT_SETTINGS, cls=completion.ExecCommand)
def exec_cmd(
    ctx: typer.Context,
    file: str | None = typer.Argument(
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
    strict_json: bool | None = typer.Option(
        None,
        "--strict-json/--no-strict-json",
        help="Require agents to return exactly one bare JSON value; default is lenient recovery.",
    ),
    max_iters: int | None = typer.Option(
        None,
        "--max-iters",
        help="Override the default do-loop iteration limit.",
    ),
    runner: str | None = typer.Option(
        None,
        "--runner",
        help="Override the default agent runner command.",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Write a structured JSONL trace log to PATH.",
        autocompletion=completion.complete_path_argument,
    ),
    no_log: bool = typer.Option(
        False,
        "--no-log",
        help="Disable trace logging.",
    ),
    _dry_run: bool = _dry_run_option(),
) -> None:
    # ``_RUN_CONTEXT_SETTINGS`` disables Click's built-in ``--help`` interception
    # (``help_option_names: []``) so that per-param ``--name`` tokens can pass
    # through to ``ctx.args``.  As a side-effect, Click may assign ``--help``/
    # ``-h`` to the optional FILE positional when they appear without a preceding
    # FILE argument.  We handle all help-trigger variants explicitly here:
    # - ``agm exec --help``  → Click assigns ``--help`` to ``file``
    # - ``agm exec -h``      → Click assigns ``-h`` to ``file``
    # - ``agm exec FILE --help`` → FILE is correct; ``--help`` lands in ctx.args
    if file in ("--help", "-h") or "--help" in ctx.args or "-h" in ctx.args:
        # When the help flag was misassigned to ``file``, treat ``file`` as absent.
        effective_file = None if file in ("--help", "-h") else file
        _exec_print_help(file=effective_file, command=command)
    del _dry_run
    if command is not None and file is not None:
        exit_with_usage_error(
            ["exec"], "error: argument FILE not allowed with -c/--command"
        )
    if command is None and file is None:
        exit_with_usage_error(
            ["exec"], "error: one of the arguments FILE -c/--command is required"
        )
    if no_log and log_file is not None:
        exit_with_usage_error(
            ["exec"], "error: --no-log and --log-file are mutually exclusive"
        )
    exec_command.run(
        ExecArgs(
            file=file,
            command=command,
            param_tokens=list(ctx.args),
            strict_json=strict_json,
            max_iters=max_iters,
            runner=runner,
            no_log=no_log,
            log_file=log_file,
        )
    )


@app.command(name="repl")
def repl_cmd(
    inputs: list[str] | None = typer.Option(
        None,
        "--input",
        help="Host input value in KEY=VALUE form (repeatable).",
    ),
    strict_json: bool | None = typer.Option(
        None,
        "--strict-json/--no-strict-json",
        help="Require agents to return exactly one bare JSON value; default is lenient recovery.",
    ),
    max_iters: int | None = typer.Option(
        None,
        "--max-iters",
        help="Override the default do-loop iteration limit.",
    ),
    runner: str | None = typer.Option(
        None,
        "--runner",
        help="Override the default agent runner command.",
    ),
    auto_agents: bool = typer.Option(
        False,
        "--auto-agents",
        help="Fire agent calls without confirming each one.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Suppress automatic echoing of entry results.",
    ),
    log_file: str | None = typer.Option(
        None,
        "--log-file",
        help="Write a structured JSONL trace log to PATH.",
        autocompletion=completion.complete_path_argument,
    ),
    no_log: bool = typer.Option(
        False,
        "--no-log",
        help="Disable trace logging.",
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    if no_log and log_file is not None:
        exit_with_usage_error(
            ["repl"], "error: --no-log and --log-file are mutually exclusive"
        )
    repl_command.run(
        ReplArgs(
            inputs=list(inputs) if inputs is not None else [],
            strict_json=strict_json,
            max_iters=max_iters,
            runner=runner,
            auto_agents=auto_agents,
            quiet=quiet,
            no_log=no_log,
            log_file=log_file,
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
    if ctx.invoked_subcommand is None:
        print_help_for_command_path(["dep"])
        raise typer.Exit()


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


@app.command()
def fetch(_help: bool = _help_option(), _dry_run: bool = _dry_run_option()) -> None:
    del _help
    del _dry_run
    fetch_command.run(object())


@app.command()
def pull(_help: bool = _help_option(), _dry_run: bool = _dry_run_option()) -> None:
    del _help
    del _dry_run
    pull_command.run(object())


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
    if no_review_file and review_file is not None:
        exit_with_usage_error(
            ["review"],
            "error: --no-review-file and --review-file are mutually exclusive",
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
        help="Maximum revision attempts. Use 'unlimited' for no limit.",
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
    if no_log and log_file is not None:
        exit_with_usage_error(
            ["refine"],
            "error: --no-log and --log-file are mutually exclusive",
        )
    if no_max_steps and max_steps is not None:
        exit_with_usage_error(
            ["refine"],
            "error: --no-max-steps and --max-steps are mutually exclusive",
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
            _parse_loop_args(
                raw_args[1:], command_path=["loop", "run"], command_optional=True
            )
        )
        return
    if raw_args[0] == "step":
        loop_step_command.run(
            _parse_loop_args(
                raw_args[1:], command_path=["loop", "step"], command_optional=True
            )
        )
        return
    loop_command.run(_parse_loop_args(raw_args, command_path=["loop"]))


@app.command()
def init(
    arg1: str | None = typer.Argument(None, metavar="arg"),
    arg2: str | None = typer.Argument(None, metavar="arg"),
    embedded: bool = typer.Option(False, "--embedded", help="Force the embedded layout."),
    workspace: bool = typer.Option(False, "--workspace", help="Force the workspace layout."),
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
    no_git_init: bool = typer.Option(
        False,
        "--no-git-init",
        help="Do not create git repositories in config/ and notes/.",
    ),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    if embedded and workspace:
        exit_with_usage_error(["init"], "error: --embedded and --workspace are mutually exclusive")
    positional: list[str] = [] if arg1 is None else [arg1] if arg2 is None else [arg1, arg2]
    init_command.run(
        InitArgs(
            positional=positional,
            branch=branch,
            embedded=embedded,
            workspace=workspace,
            clone=clone,
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


@app.command(name="list")
def list_cmd(
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show worktree directories."),
    _help: bool = _help_option(),
    _dry_run: bool = _dry_run_option(),
) -> None:
    del _help
    del _dry_run
    list_command.run(verbose=verbose)


app.add_typer(config_app, name="config")
app.add_typer(worktree_app, name="wt")
app.add_typer(worktree_app, name="worktree")
app.add_typer(dep_app, name="dep")
app.add_typer(tmux_app, name="tmux")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
