"""Shared helpers for loop commands."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agm.agent.prompt import preprocess_prompt_file
from agm.agent.prompt_source import PromptSourceOptions, resolve_prompt_source
from agm.agent.response import last_response_line
from agm.agent.runner import (
    prepare_prompt_from_source,
    split_command,
    validate_command,
)
from agm.commands.args import LoopArgs, LoopNextArgs
from agm.config.context import current_config_context
from agm.config.general import LoopConfig, load_loop_config, resolve_default_prompt_file
from agm.core.fs import is_file

LoopCommandArgs = LoopArgs | LoopNextArgs


@dataclass(slots=True)
class PreparedSelectInvocation:
    source_prompt_file: Path
    effective_prompt_file: Path
    command: list[str]
    command_kind: str
    runner_command: list[str]
    selector_command: list[str] | None


def prompt_file(filename: str) -> Path:
    return resolve_default_prompt_file(filename, home=current_config_context().home)


def configured_loop_settings(command_name: str | None) -> LoopConfig:
    context = current_config_context()
    return load_loop_config(
        home=context.home,
        proj_dir=context.proj_dir,
        cwd=context.cwd,
        command_name=command_name,
    )


def step_header_text(step: int) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = f"Step {step}  ({now})"
    sep = "-" * 61
    return (
        "\n"
        f"{sep}\n"
        f"{label.center(61)}\n"
        f"{sep}\n"
        "\n"
    )


def selected_task_text(task_file: Path) -> str:
    return f"Selected task: {task_file}\n\n"


def runner_command(args: LoopCommandArgs) -> list[str]:
    configured = configured_loop_settings(args.command_name)
    runner = args.runner if args.runner is not None else configured.runner
    selected = runner if runner is not None else "claude -p"
    return [*split_command(selected, kind="runner"), *args.runner_args]


def selector_command(args: LoopCommandArgs) -> list[str] | None:
    configured = configured_loop_settings(args.command_name)
    selector = args.selector if args.selector is not None else configured.selector
    if selector is None:
        return None
    return split_command(selector, kind="selector")


def resolved_timeout(args: LoopCommandArgs) -> float | None:
    """Resolve the idle timeout from CLI args and config."""
    if args.timeout is not None:
        return args.timeout
    configured = configured_loop_settings(args.command_name)
    return configured.timeout


def use_selector_mode(args: LoopCommandArgs) -> bool:
    """Return True when selector-based mode should be used.

    Selector mode is the default. ``--no-selector`` on the CLI or
    ``no_selector = true`` in config.toml disables it.
    """
    configured = configured_loop_settings(args.command_name)
    if args.no_selector or configured.no_selector:
        return False
    return True


def tasks_dir(args: LoopCommandArgs) -> Path:
    configured_tasks_dir = configured_loop_settings(args.command_name).tasks_dir
    selected = args.tasks_dir if args.tasks_dir is not None else configured_tasks_dir
    cwd = current_config_context().cwd
    if selected is None:
        return cwd / ".agent-files" / "tasks"

    resolved_tasks_dir = Path(selected)
    if resolved_tasks_dir.is_absolute():
        return resolved_tasks_dir
    return cwd / resolved_tasks_dir


def progress_file(args: LoopCommandArgs) -> Path:
    return tasks_dir(args) / "PROGRESS.md"


def loop_prompt_source(args: LoopCommandArgs) -> str | Path | None:
    configured = configured_loop_settings(args.command_name)
    return resolve_prompt_source(
        PromptSourceOptions(
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            config_prompt=configured.prompt,
            config_prompt_file=configured.prompt_file,
        ),
        cwd=current_config_context().cwd,
    )


def selector_prompt_source(args: LoopCommandArgs) -> str | Path | None:
    configured = configured_loop_settings(args.command_name)
    return resolve_prompt_source(
        PromptSourceOptions(
            prompt=args.selector_prompt,
            prompt_file=args.selector_prompt_file,
            config_prompt=configured.selector_prompt,
            config_prompt_file=configured.selector_prompt_file,
        ),
        cwd=current_config_context().cwd,
    )


def extra_prompt_source(args: LoopCommandArgs) -> str | Path | None:
    configured = configured_loop_settings(args.command_name)
    return resolve_prompt_source(
        PromptSourceOptions(
            prompt=args.extra_prompt,
            prompt_file=args.extra_prompt_file,
            config_prompt=configured.extra_prompt,
            config_prompt_file=configured.extra_prompt_file,
        ),
        cwd=current_config_context().cwd,
    )


def extra_selector_prompt_source(args: LoopCommandArgs) -> str | Path | None:
    configured = configured_loop_settings(args.command_name)
    return resolve_prompt_source(
        PromptSourceOptions(
            prompt=args.extra_selector_prompt,
            prompt_file=args.extra_selector_prompt_file,
            config_prompt=configured.extra_selector_prompt,
            config_prompt_file=configured.extra_selector_prompt_file,
        ),
        cwd=current_config_context().cwd,
    )


def dry_run_prompt_text(source_file: Path, effective_file: Path) -> str:
    if source_file == effective_file:
        return str(source_file)
    return f"{source_file} -> {effective_file} (preprocessed)"


def loop_env(tasks_dir: Path, *, task_file: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["TASKS_DIR"] = str(tasks_dir)
    if task_file is not None:
        env["TASK_FILE"] = str(task_file)
    return env


def prepare_select_invocation(
    args: LoopCommandArgs,
    *,
    temp_files: list[Path],
    env: dict[str, str],
) -> PreparedSelectInvocation:
    resolved_runner_command = runner_command(args)
    resolved_selector_command = selector_command(args)
    resolved_command = resolved_selector_command
    command_kind = "selector"
    if resolved_command is None:
        resolved_command = resolved_runner_command
        command_kind = "runner"
    validate_command(resolved_command, kind=command_kind)

    resolved_selector_prompt_source = selector_prompt_source(args)
    if resolved_selector_prompt_source is not None:
        resolved = prepare_prompt_from_source(
            resolved_selector_prompt_source, temp_files=temp_files, env=env
        )
        source_prompt_file = (
            resolved.source if isinstance(resolved.source, Path) else resolved.effective_file
        )
        effective_prompt_file = resolved.effective_file
    else:
        source_prompt_file = prompt_file("select.md")
        if not is_file(source_prompt_file):
            print(f"Error: prompt file not found: {source_prompt_file}", file=sys.stderr)
            raise SystemExit(1)
        effective_prompt_file = preprocess_prompt_file(
            source_prompt_file,
            temp_files=temp_files,
            env=env,
        )
    return PreparedSelectInvocation(
        source_prompt_file=source_prompt_file,
        effective_prompt_file=effective_prompt_file,
        command=resolved_command,
        command_kind=command_kind,
        runner_command=resolved_runner_command,
        selector_command=resolved_selector_command,
    )


def is_complete_output(output: str) -> bool:
    return "".join(last_response_line(output).split()) == "COMPLETE"


def selector_result(output: str, *, tasks_dir: Path) -> Path | None | str:
    selected = last_response_line(output)
    if not selected:
        return ""
    if is_complete_output(output):
        return None

    task_path = Path(selected)
    if task_path.is_absolute():
        if is_file(task_path):
            return task_path
        return selected

    resolved_task_path = current_config_context().cwd / task_path
    if is_file(resolved_task_path):
        return resolved_task_path
    tasks_dir_task_path = tasks_dir / task_path
    if is_file(tasks_dir_task_path):
        return tasks_dir_task_path
    return selected
