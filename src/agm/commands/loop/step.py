"""agm loop step."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from agm.agent.loop import (
    PreparedSelectInvocation,
    dry_run_prompt_text,
    extra_prompt_source,
    extra_selector_prompt_source,
    is_complete_output,
    loop_env,
    loop_prompt_source,
    prepare_select_invocation,
    progress_file,
    prompt_file,
    resolved_timeout,
    runner_command,
    selected_task_text,
    selector_result,
    step_header_text,
    tasks_dir,
    use_selector_mode,
)
from agm.agent.prompt import preprocess_prompt_file
from agm.agent.runner import (
    ResolvedPrompt,
    append_extra_prompt,
    cleanup_temp_files,
    command_with_prompt_target,
    prepare_prompt_from_source,
    run_prompt_command,
    validate_command,
)
from agm.cli_support.args import LoopArgs
from agm.core import dry_run
from agm.core.fs import is_file
from agm.core.log import append_log, prepare_log_file, resolve_log_file


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
    select_invocation: PreparedSelectInvocation | None
    implement_prompt_file: Path | None
    loop_prompt: PreparedPrompt | None
    resolved_prompt: ResolvedPrompt | None
    bootstrap_prompt: PreparedPrompt | None
    extra_prompt_source: str | Path | None
    log_file: Path | None
    idle_timeout: float | None


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


def _print_dry_run_command(label: str, command: list[str]) -> None:
    dry_run.print_labeled_command(label, command)


def _print_dry_run_prompt(label: str, prompt_text: str) -> None:
    print(f"dry-run: prompt [{label}]: {prompt_text}")


def prepare_runtime(args: LoopArgs) -> LoopStepRuntime:
    temp_files: list[Path] = []
    resolved_tasks_dir = tasks_dir(args)
    resolved_progress_file = progress_file(args)

    env = loop_env(resolved_tasks_dir)

    prompt_source = loop_prompt_source(args)
    resolved_prompt: ResolvedPrompt | None = None
    if prompt_source is not None:
        resolved_prompt = prepare_prompt_from_source(
            prompt_source, temp_files=temp_files, env=env
        )
    resolved_runner_command = runner_command(args)
    validate_command(resolved_runner_command, kind="runner")
    implement_prompt_file: Path | None = None
    select_invocation: PreparedSelectInvocation | None = None
    if use_selector_mode(args):
        select_invocation = prepare_select_invocation(args, temp_files=temp_files, env=env)
        if resolved_prompt is None:
            implement_prompt_file = prompt_file("implement.md")
            if not is_file(implement_prompt_file):
                print(
                    f"Error: prompt file not found: {implement_prompt_file}",
                    file=sys.stderr,
                )
                raise SystemExit(1)

    loop_prompt: PreparedPrompt | None = None
    if resolved_prompt is not None:
        loop_prompt = PreparedPrompt(
            label="prompt",
            source_file=(
                resolved_prompt.source
                if isinstance(resolved_prompt.source, Path)
                else resolved_prompt.effective_file
            ),
            effective_file=resolved_prompt.effective_file,
        )
    elif select_invocation is None:
        loop_prompt_file = prompt_file("loop.md")
        if not is_file(loop_prompt_file):
            print(f"Error: prompt file not found: {loop_prompt_file}", file=sys.stderr)
            raise SystemExit(1)
        loop_prompt = _prepare_prompt("loop", loop_prompt_file, temp_files=temp_files, env=env)

    bootstrap_prompt: PreparedPrompt | None = None
    if select_invocation is None and not is_file(resolved_progress_file):
        bootstrap_prompt_file = prompt_file("select.md")
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
            run_prompt_command(
                resolved_runner_command,
                bootstrap_prompt.effective_file,
                env=env,
                idle_timeout=resolved_timeout(args),
            )

    resolved_extra_prompt_source = extra_prompt_source(args)
    resolved_extra_selector_prompt_source = extra_selector_prompt_source(args)

    # Apply extra selector prompt to the selector invocation
    if (
        select_invocation is not None
        and resolved_extra_selector_prompt_source is not None
    ):
        new_effective = append_extra_prompt(
            select_invocation.effective_prompt_file,
            resolved_extra_selector_prompt_source,
            temp_files=temp_files,
            env=env,
        )
        select_invocation.effective_prompt_file = new_effective

    # Apply extra prompt to the loop prompt (no-selector mode)
    if loop_prompt is not None and resolved_extra_prompt_source is not None:
        new_effective = append_extra_prompt(
            loop_prompt.effective_file,
            resolved_extra_prompt_source,
            temp_files=temp_files,
            env=env,
        )
        loop_prompt.effective_file = new_effective

    log_file = resolve_log_file(
        command_name="loop",
        enabled=not args.no_log,
        log_file=args.log_file,
    )
    prepare_log_file(log_file)

    timeout = resolved_timeout(args)

    return LoopStepRuntime(
        temp_files=temp_files,
        resolved_tasks_dir=resolved_tasks_dir,
        resolved_progress_file=resolved_progress_file,
        env=env,
        resolved_runner_command=resolved_runner_command,
        select_invocation=select_invocation,
        implement_prompt_file=implement_prompt_file,
        loop_prompt=loop_prompt,
        resolved_prompt=resolved_prompt,
        bootstrap_prompt=bootstrap_prompt,
        extra_prompt_source=resolved_extra_prompt_source,
        log_file=log_file,
        idle_timeout=timeout,
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
    dry_run.print_detail(
        "idle timeout",
        f"{runtime.idle_timeout}s" if runtime.idle_timeout is not None else "disabled",
    )
    has_selector_command = (
        runtime.select_invocation is not None
        and runtime.select_invocation.selector_command is not None
    )
    selector_command_text = "disabled"
    if has_selector_command:
        assert runtime.select_invocation is not None
        assert runtime.select_invocation.selector_command is not None
        selector_command_text = dry_run.format_command(
            runtime.select_invocation.selector_command
        )
    dry_run.print_detail("selector command", selector_command_text)

    prompts = [runtime.bootstrap_prompt, runtime.loop_prompt]
    for prompt in prompts:
        if prompt is None:
            continue
        _print_dry_run_prompt(
            prompt.label,
            dry_run_prompt_text(prompt.source_file, prompt.effective_file),
        )
    if runtime.select_invocation is not None:
        _print_dry_run_prompt(
            "selector",
            dry_run_prompt_text(
                runtime.select_invocation.source_prompt_file,
                runtime.select_invocation.effective_prompt_file,
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

    if runtime.select_invocation is None:
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
        if runtime.resolved_prompt is not None:
            dry_run.print_detail("explicit prompt", str(runtime.resolved_prompt.effective_file))
        return

    _print_dry_run_command(
        "selector",
        command_with_prompt_target(
            runtime.select_invocation.command,
            runtime.select_invocation.effective_prompt_file,
        ),
    )
    if runtime.resolved_prompt is not None:
        dry_run.print_detail("runner prompt", str(runtime.resolved_prompt.effective_file))
    elif runtime.implement_prompt_file is not None:
        dry_run.print_detail(
            "runner prompt", f"{runtime.implement_prompt_file} (default)"
        )
    dry_run.print_operation(
        "loop-runner",
        "subsequent runner invocations depend on selector output",
    )


def print_startup(runtime: LoopStepRuntime) -> None:
    message = f"Tasks dir: {runtime.resolved_tasks_dir}\n"
    print(message, end="")
    append_log(runtime.log_file, message)


def execute_single_step(runtime: LoopStepRuntime, *, step_number: int) -> bool:
    header = step_header_text(step_number)
    print(header, end="")
    append_log(runtime.log_file, header)

    def stdout_callback(chunk: str) -> None:
        append_log(runtime.log_file, chunk)
        _write_stream(chunk)

    def stderr_callback(chunk: str) -> None:
        append_log(runtime.log_file, chunk)
        _write_stream(chunk, stderr=True)

    if runtime.select_invocation is None:
        assert runtime.loop_prompt is not None
        output = run_prompt_command(
            runtime.resolved_runner_command,
            runtime.loop_prompt.effective_file,
            env=runtime.env,
            stdout_callback=stdout_callback,
            stderr_callback=stderr_callback,
            idle_timeout=runtime.idle_timeout,
        )
        if is_complete_output(output):
            print("\nCompleted.")
            return True
        return False

    while True:
        selector_output = run_prompt_command(
            runtime.select_invocation.command,
            runtime.select_invocation.effective_prompt_file,
            env=runtime.env,
            stdout_callback=stdout_callback,
            stderr_callback=stderr_callback,
            idle_timeout=runtime.idle_timeout,
        )
        next_task = selector_result(selector_output, tasks_dir=runtime.resolved_tasks_dir)
        if next_task is None:
            print("\nCompleted.")
            return True
        if isinstance(next_task, Path):
            break

    selected_task_output = selected_task_text(next_task)
    append_log(runtime.log_file, "\n" + selected_task_output)
    _write_stream("\n" + selected_task_output)

    if runtime.resolved_prompt is not None:
        runner_env = loop_env(runtime.resolved_tasks_dir, task_file=next_task)
        # Re-prepare the prompt from the original source so that env vars
        # like ${TASK_FILE} (which are only available after task selection) are
        # expanded correctly.
        re_resolved = prepare_prompt_from_source(
            runtime.resolved_prompt.source,
            temp_files=runtime.temp_files,
            env=runner_env,
        )
        runner_target = re_resolved.effective_file
        if runtime.extra_prompt_source is not None:
            runner_target = append_extra_prompt(
                runner_target,
                runtime.extra_prompt_source,
                temp_files=runtime.temp_files,
                env=runner_env,
            )
    elif runtime.implement_prompt_file is not None:
        runner_env = loop_env(runtime.resolved_tasks_dir, task_file=next_task)
        runner_target = preprocess_prompt_file(
            runtime.implement_prompt_file,
            temp_files=runtime.temp_files,
            env=runner_env,
        )
        if runtime.extra_prompt_source is not None:
            runner_target = append_extra_prompt(
                runner_target,
                runtime.extra_prompt_source,
                temp_files=runtime.temp_files,
                env=runner_env,
            )
    else:
        runner_env = runtime.env
        runner_target = next_task

    run_prompt_command(
        runtime.resolved_runner_command,
        runner_target,
        env=runner_env,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback,
        idle_timeout=runtime.idle_timeout,
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
        print_startup(runtime)
        execute_single_step(runtime, step_number=1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
    finally:
        if runtime is not None:
            cleanup_runtime(runtime)
