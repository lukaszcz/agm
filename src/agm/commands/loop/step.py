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
    PreparedProgressInvocation,
    cleanup_temp_files,
    command_with_prompt_target,
    is_complete_output,
    loop_env,
    prepare_progress_invocation,
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
class PreparedPrompt:
    label: str
    source_file: Path
    effective_file: Path


@dataclass(slots=True)
class LoopStepRuntime:
    temp_files: list[Path]
    resolved_tasks_dir: Path
    resolved_progress_file: Path
    env: dict[str, str]
    resolved_runner_command: list[str]
    progress_invocation: PreparedProgressInvocation | None
    loop_prompt: PreparedPrompt | None
    bootstrap_prompt: PreparedPrompt | None
    log_file: Path | None


def _log_file(args: LoopArgs) -> Path | None:
    if args.no_log:
        return None
    if args.log_file is not None:
        return Path(args.log_file)
    return Path.cwd() / f"loop-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"


def _append_log(log_file: Path | None, content: str) -> None:
    if log_file is None or not content:
        return
    append_text(log_file, content, encoding="utf-8")


def _write_stream(chunk: str, *, stderr: bool = False) -> None:
    if not chunk:
        return
    stream = sys.stderr if stderr else sys.stdout
    stream.write(chunk)
    stream.flush()


def _prepare_prompt(
    prompt_label: str,
    prompt_source_file: Path,
    *,
    temp_files: list[Path],
    env: dict[str, str],
) -> PreparedPrompt:
    return PreparedPrompt(
        label=prompt_label,
        source_file=prompt_source_file,
        effective_file=preprocess_prompt_file(prompt_source_file, temp_files=temp_files, env=env),
    )


def _dry_run_prompt_text(prompt: PreparedPrompt) -> str:
    if prompt.source_file == prompt.effective_file:
        return str(prompt.source_file)
    return f"{prompt.source_file} -> {prompt.effective_file} (preprocessed)"


def _print_dry_run_command(label: str, command: list[str]) -> None:
    dry_run.print_labeled_command(label, command)


def _print_dry_run_prompt(label: str, prompt_text: str) -> None:
    print(f"dry-run: prompt [{label}]: {prompt_text}")


def prepare_runtime(args: LoopArgs) -> LoopStepRuntime:
    temp_files: list[Path] = []
    resolved_tasks_dir = tasks_dir(args)
    resolved_progress_file = progress_file(args)
    env = loop_env(resolved_tasks_dir)
    resolved_runner_command = runner_command(args)
    validate_command(resolved_runner_command, kind="runner")
    resolved_selector_command = selector_command(args)
    progress_invocation: PreparedProgressInvocation | None = None
    if resolved_selector_command is not None:
        progress_invocation = prepare_progress_invocation(args, temp_files=temp_files, env=env)

    loop_prompt: PreparedPrompt | None = None
    if progress_invocation is None:
        loop_prompt_file = prompt_file("loop.md")
        if not is_file(loop_prompt_file):
            print(f"Error: prompt file not found: {loop_prompt_file}", file=sys.stderr)
            raise SystemExit(1)
        loop_prompt = _prepare_prompt("loop", loop_prompt_file, temp_files=temp_files, env=env)

    bootstrap_prompt: PreparedPrompt | None = None
    if progress_invocation is None and not is_file(resolved_progress_file):
        bootstrap_prompt_file = prompt_file("update_progress.md")
        if not is_file(bootstrap_prompt_file):
            print(f"Error: prompt file not found: {bootstrap_prompt_file}", file=sys.stderr)
            raise SystemExit(1)
        bootstrap_prompt = _prepare_prompt(
            "bootstrap",
            bootstrap_prompt_file,
            temp_files=temp_files,
            env=env,
        )
        if not dry_run.enabled():
            run_command(resolved_runner_command, bootstrap_prompt.effective_file, env=env)

    log_file = _log_file(args)
    if log_file is not None:
        print(f"Logging to {log_file if args.log_file is not None else log_file.name}")
        mkdir(log_file.parent, parents=True, exist_ok=True)

    return LoopStepRuntime(
        temp_files=temp_files,
        resolved_tasks_dir=resolved_tasks_dir,
        resolved_progress_file=resolved_progress_file,
        env=env,
        resolved_runner_command=resolved_runner_command,
        progress_invocation=progress_invocation,
        loop_prompt=loop_prompt,
        bootstrap_prompt=bootstrap_prompt,
        log_file=log_file,
    )


def print_dry_run(runtime: LoopStepRuntime) -> None:
    dry_run.print_configuration("loop")
    dry_run.print_detail("tasks dir", str(runtime.resolved_tasks_dir))
    dry_run.print_detail("progress file", str(runtime.resolved_progress_file))
    dry_run.print_detail(
        "log file",
        str(runtime.log_file) if runtime.log_file is not None else "disabled",
    )
    dry_run.print_detail("runner command", dry_run.format_command(runtime.resolved_runner_command))
    has_selector_command = (
        runtime.progress_invocation is not None
        and runtime.progress_invocation.selector_command is not None
    )
    selector_command_text = "disabled"
    if has_selector_command:
        assert runtime.progress_invocation is not None
        assert runtime.progress_invocation.selector_command is not None
        selector_command_text = dry_run.format_command(
            runtime.progress_invocation.selector_command
        )
    dry_run.print_detail("selector command", selector_command_text)

    prompts = [runtime.bootstrap_prompt, runtime.loop_prompt]
    for prompt in prompts:
        if prompt is None:
            continue
        _print_dry_run_prompt(prompt.label, _dry_run_prompt_text(prompt))
    if runtime.progress_invocation is not None:
        _print_dry_run_prompt(
            "selector",
            _dry_run_prompt_text(
                PreparedPrompt(
                    label="selector",
                    source_file=runtime.progress_invocation.source_prompt_file,
                    effective_file=runtime.progress_invocation.effective_prompt_file,
                )
            ),
        )

    if runtime.bootstrap_prompt is not None:
        _print_dry_run_command(
            "bootstrap",
            command_with_prompt_target(
                runtime.resolved_runner_command,
                runtime.bootstrap_prompt.effective_file,
            ),
        )

    if runtime.progress_invocation is None:
        assert runtime.loop_prompt is not None
        _print_dry_run_command(
            "runner",
            command_with_prompt_target(
                runtime.resolved_runner_command,
                runtime.loop_prompt.effective_file,
            ),
        )
        dry_run.print_operation(
            "loop-runner",
            "runner command repeats until output is COMPLETE",
        )
        return

    _print_dry_run_command(
        "selector",
        command_with_prompt_target(
            runtime.progress_invocation.command,
            runtime.progress_invocation.effective_prompt_file,
        ),
    )
    dry_run.print_operation(
        "loop-runner",
        "subsequent runner invocations depend on selector output",
    )


def execute_single_step(runtime: LoopStepRuntime, *, step_number: int) -> bool:
    header = step_header_text(step_number)
    print(header, end="")
    _append_log(runtime.log_file, header)

    def stdout_callback(chunk: str) -> None:
        _append_log(runtime.log_file, chunk)
        _write_stream(chunk)

    def stderr_callback(chunk: str) -> None:
        _append_log(runtime.log_file, chunk)
        _write_stream(chunk, stderr=True)

    if runtime.progress_invocation is None:
        assert runtime.loop_prompt is not None
        output = run_command(
            runtime.resolved_runner_command,
            runtime.loop_prompt.effective_file,
            env=runtime.env,
            stdout_callback=stdout_callback,
            stderr_callback=stderr_callback,
        )
        if is_complete_output(output):
            print("\nCompleted.")
            return True
        return False

    while True:
        selector_output = run_command(
            runtime.progress_invocation.command,
            runtime.progress_invocation.effective_prompt_file,
            env=runtime.env,
            stdout_callback=stdout_callback,
            stderr_callback=stderr_callback,
        )
        next_task = selector_result(selector_output, tasks_dir=runtime.resolved_tasks_dir)
        if next_task is None:
            print("\nCompleted.")
            return True
        if isinstance(next_task, Path):
            break

    selected_task_output = selected_task_text(next_task)
    _append_log(runtime.log_file, "\n" + selected_task_output)
    _write_stream("\n" + selected_task_output)
    run_command(
        runtime.resolved_runner_command,
        next_task,
        env=runtime.env,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback,
    )
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
