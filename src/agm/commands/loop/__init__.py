"""agm loop."""

from __future__ import annotations

import sys
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


def run(args: LoopArgs) -> None:
    temp_files: list[Path] = []
    resolved_tasks_dir = tasks_dir(args)
    env = loop_env(resolved_tasks_dir)
    resolved_runner_command = runner_command(args)
    resolved_selector_command = selector_command(args)
    validate_command(resolved_runner_command, kind="runner")
    if resolved_selector_command is not None:
        validate_command(resolved_selector_command, kind="selector")

    try:
        loop_prompt_file: Path | None = None
        if resolved_selector_command is None:
            loop_prompt_file = prompt_file("loop.md")
            if not is_file(loop_prompt_file):
                print(f"Error: prompt file not found: {loop_prompt_file}", file=sys.stderr)
                raise SystemExit(1)
            loop_prompt_file = preprocess_prompt_file(
                loop_prompt_file, temp_files=temp_files, env=env
            )

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

        if dry_run.enabled():
            if resolved_selector_command is None:
                target = (
                    loop_prompt_file if loop_prompt_file is not None else prompt_file("loop.md")
                )
                dry_run.print_command(command_with_prompt_target(resolved_runner_command, target))
            else:
                assert selector_prompt_file is not None
                selector_args = command_with_prompt_target(
                    resolved_selector_command,
                    selector_prompt_file,
                )
                dry_run.print_command(selector_args)
                dry_run.print_operation(
                    "loop-runner",
                    "subsequent runner invocations depend on selector output",
                )
            return

        step = 1
        while True:
            header = step_header_text(step)
            print(header, end="")
            step += 1

            if resolved_selector_command is None:
                assert loop_prompt_file is not None
                output = run_command(resolved_runner_command, loop_prompt_file, env=env)
                _append_log(log_file, header, output)
                _print_output(output)
                if "".join(output.split()) == "COMPLETE":
                    print("\nCompleted.")
                    break
                continue

            assert selector_prompt_file is not None
            selector_outputs: list[str] = []
            while True:
                selector_output = run_command(
                    resolved_selector_command,
                    selector_prompt_file,
                    env=env,
                )
                selector_outputs.append(selector_output)

                next_task = selector_result(selector_output, tasks_dir=resolved_tasks_dir)
                if next_task is None:
                    combined_selector_output = "".join(selector_outputs)
                    _append_log(log_file, header, combined_selector_output)
                    _print_output(combined_selector_output)
                    print("\nCompleted.")
                    return
                if isinstance(next_task, Path):
                    break

            selected_task_output = selected_task_text(next_task)
            selector_transcript = "".join(selector_outputs)
            _append_log(log_file, header, selected_task_output + selector_transcript)
            _print_output(selector_transcript + "\n" + selected_task_output)
            runner_output = run_command(resolved_runner_command, next_task, env=env)
            _append_log(log_file, "", runner_output)
            _print_output(runner_output)
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
    finally:
        cleanup_temp_files(temp_files)
