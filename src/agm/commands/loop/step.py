"""agm loop step."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agm.commands.args import LoopArgs
from agm.core import dry_run
from agm.core.fs import append_text, is_file, mkdir
from agm.core.prompt import preprocess_prompt_file

from .common import (
    cleanup_temp_files,
    command_with_prompt_target,
    loop_env,
    progress_file,
    prompt_file,
    run_command,
    runner_command,
    selected_task_text,
    selector_command,
    selector_result,
    step_header_text,
    tasks_dir,
    validate_command,
)


@dataclass(slots=True)
class LoopStepRuntime:
    temp_files: list[Path]
    resolved_tasks_dir: Path
    env: dict[str, str]
    resolved_runner_command: list[str]
    resolved_selector_command: list[str] | None
    loop_prompt_file: Path | None
    selector_prompt_file: Path | None
    log_file: Path | None


def _log_file(args: LoopArgs) -> Path | None:
    if args.no_log:
        return None
    if args.log_file is not None:
        return Path(args.log_file)
    return Path.cwd() / f"loop-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"


def _append_log(log_file: Path | None, header: str, output: str) -> None:
    if log_file is None:
        return
    append_text(log_file, header + output, encoding="utf-8")


def _print_output(output: str) -> None:
    if output:
        print(output, end="")


def prepare_runtime(args: LoopArgs) -> LoopStepRuntime:
    temp_files: list[Path] = []
    resolved_tasks_dir = tasks_dir(args)
    env = loop_env(resolved_tasks_dir)
    resolved_runner_command = runner_command(args)
    resolved_selector_command = selector_command(args)
    validate_command(resolved_runner_command, kind="runner")
    if resolved_selector_command is not None:
        validate_command(resolved_selector_command, kind="selector")

    loop_prompt_file: Path | None = None
    if resolved_selector_command is None:
        loop_prompt_file = prompt_file("loop.md")
        if not is_file(loop_prompt_file):
            print(f"Error: prompt file not found: {loop_prompt_file}", file=sys.stderr)
            raise SystemExit(1)
        loop_prompt_file = preprocess_prompt_file(loop_prompt_file, temp_files=temp_files, env=env)

    resolved_progress_file = progress_file(args)
    if resolved_selector_command is None and not is_file(resolved_progress_file):
        bootstrap_prompt_file = prompt_file("update_progress.md")
        if not is_file(bootstrap_prompt_file):
            print(f"Error: prompt file not found: {bootstrap_prompt_file}", file=sys.stderr)
            raise SystemExit(1)
        bootstrap_prompt_file = preprocess_prompt_file(
            bootstrap_prompt_file, temp_files=temp_files, env=env
        )
        run_command(resolved_runner_command, bootstrap_prompt_file, env=env)

    selector_prompt_file: Path | None = None
    if resolved_selector_command is not None:
        selector_prompt_file = prompt_file("update_progress.md")
        if not is_file(selector_prompt_file):
            print(f"Error: prompt file not found: {selector_prompt_file}", file=sys.stderr)
            raise SystemExit(1)
        selector_prompt_file = preprocess_prompt_file(
            selector_prompt_file, temp_files=temp_files, env=env
        )

    log_file = _log_file(args)
    if log_file is not None:
        print(f"Logging to {log_file if args.log_file is not None else log_file.name}")
        mkdir(log_file.parent, parents=True, exist_ok=True)

    return LoopStepRuntime(
        temp_files=temp_files,
        resolved_tasks_dir=resolved_tasks_dir,
        env=env,
        resolved_runner_command=resolved_runner_command,
        resolved_selector_command=resolved_selector_command,
        loop_prompt_file=loop_prompt_file,
        selector_prompt_file=selector_prompt_file,
        log_file=log_file,
    )


def print_dry_run(runtime: LoopStepRuntime) -> None:
    if runtime.resolved_selector_command is None:
        target = (
            runtime.loop_prompt_file
            if runtime.loop_prompt_file is not None
            else prompt_file("loop.md")
        )
        dry_run.print_command(command_with_prompt_target(runtime.resolved_runner_command, target))
        return

    assert runtime.selector_prompt_file is not None
    dry_run.print_command(
        command_with_prompt_target(runtime.resolved_selector_command, runtime.selector_prompt_file)
    )


def execute_single_step(runtime: LoopStepRuntime, *, step_number: int) -> bool:
    header = step_header_text(step_number)
    print(header, end="")

    if runtime.resolved_selector_command is None:
        assert runtime.loop_prompt_file is not None
        output = run_command(
            runtime.resolved_runner_command,
            runtime.loop_prompt_file,
            env=runtime.env,
        )
        _append_log(runtime.log_file, header, output)
        _print_output(output)
        if "".join(output.split()) == "COMPLETE":
            print("\nCompleted.")
            return True
        return False

    assert runtime.selector_prompt_file is not None
    selector_outputs: list[str] = []
    while True:
        selector_output = run_command(
            runtime.resolved_selector_command,
            runtime.selector_prompt_file,
            env=runtime.env,
        )
        selector_outputs.append(selector_output)

        next_task = selector_result(selector_output, tasks_dir=runtime.resolved_tasks_dir)
        if next_task is None:
            combined_selector_output = "".join(selector_outputs)
            _append_log(runtime.log_file, header, combined_selector_output)
            _print_output(combined_selector_output)
            print("\nCompleted.")
            return True
        if isinstance(next_task, Path):
            break

    selected_task_output = selected_task_text(next_task)
    selector_transcript = "".join(selector_outputs)
    _append_log(runtime.log_file, header, selected_task_output + selector_transcript)
    _print_output(selector_transcript + "\n" + selected_task_output)
    runner_output = run_command(runtime.resolved_runner_command, next_task, env=runtime.env)
    _append_log(runtime.log_file, "", runner_output)
    _print_output(runner_output)
    return False


def cleanup_runtime(runtime: LoopStepRuntime) -> None:
    cleanup_temp_files(runtime.temp_files)


def run(args: LoopArgs) -> None:
    runtime: LoopStepRuntime | None = None
    try:
        runtime = prepare_runtime(args)
        if dry_run.enabled():
            print_dry_run(runtime)
            return
        execute_single_step(runtime, step_number=1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
    finally:
        if runtime is not None:
            cleanup_runtime(runtime)
