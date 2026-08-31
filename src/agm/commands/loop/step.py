"""agm loop step."""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO, cast

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
from agm.agent.prompt import (
    preprocess_prompt_file,
    prompt_source_label,
    require_prompt_file,
    validate_prompt_template,
)
from agm.agent.runner import (
    AgentCallTimeout,
    append_extra_prompt,
    cleanup_temp_files,
    command_with_prompt_target_or_exit,
    prepare_prompt_from_source,
    run_prompt_command,
    validate_command,
)
from agm.cli_support.args import LoopArgs
from agm.core import dry_run
from agm.core.fs import is_file
from agm.core.log import append_log, prepare_log_file, resolve_log_file
from agm.core.path import display_path


@dataclass(slots=True)
class PreparedPrompt:
    label: str
    source_file: Path
    effective_file: Path


@dataclass(slots=True)
class SelectorStep:
    """A prepared selector invocation and the runner prompt it feeds.

    Selector mode always resolves exactly one runner prompt source — the
    configured prompt, or the default ``implement.md`` — so the two travel
    together and the post-selection runner has a single prompt to render.
    """

    invocation: PreparedSelectInvocation
    runner_prompt_source: str | Path


@dataclass(slots=True)
class LoopStepRuntime:
    temp_files: list[Path]
    resolved_tasks_dir: Path
    resolved_progress_file: Path
    env: dict[str, str]
    resolved_runner_command: list[str]
    selector: SelectorStep | None
    loop_prompt: PreparedPrompt | None
    prompt_source: str | Path | None
    bootstrap_prompt: PreparedPrompt | None
    extra_prompt_source: str | Path | None
    log_file: Path | None
    idle_timeout: float | None


class _DiagnosticLog:
    """Mirror diagnostics to the terminal and the loop log."""

    def __init__(self, stream: TextIO, log_file: Path | None) -> None:
        self._stream = stream
        self._log_file = log_file

    def write(self, text: str) -> int:
        self._stream.write(text)
        append_log(self._log_file, text)
        return len(text)


@contextmanager
def _record_diagnostics(log_file: Path | None) -> Iterator[None]:
    with redirect_stderr(cast(TextIO, _DiagnosticLog(sys.stderr, log_file))):
        yield


def _write_stream(chunk: str, *, stderr: bool = False) -> None:
    if not chunk:
        return
    stream = sys.stderr if stderr else sys.stdout
    stream.write(chunk)
    stream.flush()


def _run_agent_call(
    command: list[str],
    target: Path,
    *,
    env: dict[str, str],
    stdout_callback: Callable[[str], None] | None = None,
    stderr_callback: Callable[[str], None] | None = None,
    idle_timeout: float | None,
) -> str | None:
    """Run one agent call, treating a timeout as a failed invocation.

    A spawn failure that ``run_prompt_command`` cannot classify — an argv the
    OS rejects outright — still reaches the caller, but is reported here first
    so it lands in the loop log alongside the agent's own output.
    """
    try:
        return run_prompt_command(
            command,
            target,
            env=env,
            stdout_callback=stdout_callback,
            stderr_callback=stderr_callback,
            idle_timeout=idle_timeout,
        )
    except AgentCallTimeout:
        return None
    except ValueError as exc:
        message = f"Error: agent call failed: {exc}\n"
        if stderr_callback is None:
            print(message, end="", file=sys.stderr)
        else:
            stderr_callback(message)
        raise


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


def _require_prompt_source(source: str | Path | None, *, label: str = "prompt") -> None:
    """Check a not-yet-prepared prompt source, which may be inline text."""
    if isinstance(source, Path):
        require_prompt_file(source, label=label)


def _prepare_runtime(args: LoopArgs, *, log_file: Path | None) -> LoopStepRuntime:
    temp_files: list[Path] = []
    resolved_tasks_dir = tasks_dir(args)
    resolved_progress_file = progress_file(args)

    env = loop_env(resolved_tasks_dir)

    prompt_source = loop_prompt_source(args)
    _require_prompt_source(prompt_source)
    resolved_extra_prompt_source = extra_prompt_source(args)
    _require_prompt_source(resolved_extra_prompt_source, label="extra prompt")

    resolved_runner_command = runner_command(args)
    validate_command(resolved_runner_command, kind="runner", env=env)
    selector_mode = use_selector_mode(args)
    selector: SelectorStep | None = None
    if selector_mode:
        # Runner prompts are rendered only after selection, but their
        # structure and names can be checked before the selector makes any
        # side effects. TASK_FILE's eventual value is immaterial here.
        if prompt_source is None:
            implement_prompt_file = prompt_file("implement.md")
            require_prompt_file(implement_prompt_file)
            runner_prompt_source: str | Path = implement_prompt_file
        else:
            runner_prompt_source = prompt_source
        runner_variables = {**env, "TASK_FILE": ""}
        validate_prompt_template(runner_prompt_source, variables=runner_variables)
        if resolved_extra_prompt_source is not None:
            validate_prompt_template(
                resolved_extra_prompt_source,
                variables=runner_variables,
                label="extra prompt",
            )
        selector = SelectorStep(
            invocation=prepare_select_invocation(args, temp_files=temp_files, env=env),
            runner_prompt_source=runner_prompt_source,
        )

    loop_prompt: PreparedPrompt | None = None
    if prompt_source is not None and not selector_mode:
        resolved_prompt = prepare_prompt_from_source(prompt_source, temp_files=temp_files, env=env)
        loop_prompt = PreparedPrompt(
            label="prompt",
            source_file=(
                resolved_prompt.source
                if isinstance(resolved_prompt.source, Path)
                else resolved_prompt.effective_file
            ),
            effective_file=resolved_prompt.effective_file,
        )
    elif selector is None:
        loop_prompt_file = prompt_file("loop.md")
        require_prompt_file(loop_prompt_file)
        loop_prompt = _prepare_prompt("loop", loop_prompt_file, temp_files=temp_files, env=env)

    bootstrap_prompt: PreparedPrompt | None = None
    if selector is None and not is_file(resolved_progress_file):
        bootstrap_prompt_file = prompt_file("select.md")
        require_prompt_file(bootstrap_prompt_file)
        bootstrap_prompt = _prepare_prompt(
            "bootstrap",
            bootstrap_prompt_file,
            temp_files=temp_files,
            env=env,
        )
        if not dry_run.enabled():
            _run_agent_call(
                resolved_runner_command,
                bootstrap_prompt.effective_file,
                env=env,
                idle_timeout=resolved_timeout(args),
            )

    resolved_extra_selector_prompt_source = extra_selector_prompt_source(args)

    # Apply extra selector prompt to the selector invocation
    if selector is not None and resolved_extra_selector_prompt_source is not None:
        new_effective = append_extra_prompt(
            selector.invocation.effective_prompt_file,
            resolved_extra_selector_prompt_source,
            temp_files=temp_files,
            env=env,
        )
        selector.invocation.effective_prompt_file = new_effective

    # Apply extra prompt to the loop prompt (no-selector mode)
    if loop_prompt is not None and resolved_extra_prompt_source is not None:
        new_effective = append_extra_prompt(
            loop_prompt.effective_file,
            resolved_extra_prompt_source,
            temp_files=temp_files,
            env=env,
        )
        loop_prompt.effective_file = new_effective

    timeout = resolved_timeout(args)

    return LoopStepRuntime(
        temp_files=temp_files,
        resolved_tasks_dir=resolved_tasks_dir,
        resolved_progress_file=resolved_progress_file,
        env=env,
        resolved_runner_command=resolved_runner_command,
        selector=selector,
        loop_prompt=loop_prompt,
        prompt_source=prompt_source,
        bootstrap_prompt=bootstrap_prompt,
        extra_prompt_source=resolved_extra_prompt_source,
        log_file=log_file,
        idle_timeout=timeout,
    )


def prepare_runtime(args: LoopArgs) -> LoopStepRuntime:
    """Prepare a loop runtime while recording all setup diagnostics."""
    log_file = resolve_log_file(
        command_name="loop",
        enabled=not args.no_log,
        log_file=args.log_file,
    )
    prepare_log_file(log_file)
    with _record_diagnostics(log_file):
        return _prepare_runtime(args, log_file=log_file)


def print_dry_run(runtime: LoopStepRuntime) -> None:
    dry_run.print_configuration("loop")
    dry_run.print_detail("tasks dir", display_path(runtime.resolved_tasks_dir))
    dry_run.print_detail("progress file", display_path(runtime.resolved_progress_file))
    dry_run.print_detail(
        "log file",
        display_path(runtime.log_file) if runtime.log_file is not None else "disabled",
    )
    dry_run.print_detail("runner command", dry_run.format_command(runtime.resolved_runner_command))
    dry_run.print_detail(
        "idle timeout",
        f"{runtime.idle_timeout}s" if runtime.idle_timeout is not None else "disabled",
    )
    selector = runtime.selector
    selector_command_text = "disabled"
    if selector is not None and selector.invocation.selector_command is not None:
        selector_command_text = dry_run.format_command(selector.invocation.selector_command)
    dry_run.print_detail("selector command", selector_command_text)

    prompts = [runtime.bootstrap_prompt, runtime.loop_prompt]
    for prompt in prompts:
        if prompt is None:
            continue
        _print_dry_run_prompt(
            prompt.label,
            dry_run_prompt_text(prompt.source_file, prompt.effective_file),
        )
    if selector is not None and runtime.prompt_source is not None:
        _print_dry_run_prompt(
            "prompt",
            prompt_source_label(runtime.prompt_source),
        )
    if selector is not None:
        _print_dry_run_prompt(
            "selector",
            dry_run_prompt_text(
                selector.invocation.source_prompt_file,
                selector.invocation.effective_prompt_file,
            ),
        )

    if runtime.bootstrap_prompt is not None:
        _print_dry_run_command(
            "bootstrap",
            command_with_prompt_target_or_exit(
                runtime.resolved_runner_command,
                runtime.bootstrap_prompt.effective_file,
                runtime.env,
            ),
        )

    if selector is None:
        assert runtime.loop_prompt is not None
        _print_dry_run_command(
            "runner",
            command_with_prompt_target_or_exit(
                runtime.resolved_runner_command,
                runtime.loop_prompt.effective_file,
                runtime.env,
            ),
        )
        dry_run.print_operation(
            "loop-runner",
            "runner command repeats until output is COMPLETE",
        )
        if runtime.prompt_source is not None:
            dry_run.print_detail(
                "explicit prompt", display_path(runtime.loop_prompt.effective_file)
            )
        return

    _print_dry_run_command(
        "selector",
        command_with_prompt_target_or_exit(
            selector.invocation.command,
            selector.invocation.effective_prompt_file,
            runtime.env,
        ),
    )
    dry_run.print_detail("TASK_FILE", "unavailable (the selector is not run in dry-run mode)")
    default_marker = "" if runtime.prompt_source is not None else "(default) "
    dry_run.print_detail(
        "runner prompt",
        f"{prompt_source_label(selector.runner_prompt_source)} "
        f"{default_marker}(reprocessed after task selection)",
    )
    dry_run.print_operation(
        "loop-runner",
        "would run after task selection; no selector or runner is executed",
    )


def print_startup(runtime: LoopStepRuntime) -> None:
    message = f"Tasks dir: {display_path(runtime.resolved_tasks_dir)}\n"
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

    selector = runtime.selector
    if selector is None:
        assert runtime.loop_prompt is not None
        output = _run_agent_call(
            runtime.resolved_runner_command,
            runtime.loop_prompt.effective_file,
            env=runtime.env,
            stdout_callback=stdout_callback,
            stderr_callback=stderr_callback,
            idle_timeout=runtime.idle_timeout,
        )
        if output is None:
            return False
        if is_complete_output(output):
            completion = "\nCompleted.\n"
            print(completion, end="")
            append_log(runtime.log_file, completion)
            return True
        return False

    while True:
        selector_output = _run_agent_call(
            selector.invocation.command,
            selector.invocation.effective_prompt_file,
            env=runtime.env,
            stdout_callback=stdout_callback,
            stderr_callback=stderr_callback,
            idle_timeout=runtime.idle_timeout,
        )
        if selector_output is None:
            continue
        next_task = selector_result(selector_output, tasks_dir=runtime.resolved_tasks_dir)
        if next_task is None:
            completion = "\nCompleted.\n"
            print(completion, end="")
            append_log(runtime.log_file, completion)
            return True
        if isinstance(next_task, Path):
            break

    selected_task_output = selected_task_text(next_task)
    append_log(runtime.log_file, "\n" + selected_task_output)
    _write_stream("\n" + selected_task_output)

    with _record_diagnostics(runtime.log_file):
        runner_env, runner_target = _runner_target(
            runtime, selector.runner_prompt_source, next_task
        )

    _run_agent_call(
        runtime.resolved_runner_command,
        runner_target,
        env=runner_env,
        stdout_callback=stdout_callback,
        stderr_callback=stderr_callback,
        idle_timeout=runtime.idle_timeout,
    )
    return False


def _runner_target(
    runtime: LoopStepRuntime,
    runner_prompt_source: str | Path,
    next_task: Path,
) -> tuple[dict[str, str], Path]:
    """Build the selected task's runner environment and prompt target."""
    runner_env = loop_env(runtime.resolved_tasks_dir, task_file=next_task)
    resolved_prompt = prepare_prompt_from_source(
        runner_prompt_source,
        temp_files=runtime.temp_files,
        env=runner_env,
    )
    runner_target = resolved_prompt.effective_file
    if runtime.extra_prompt_source is not None:
        runner_target = append_extra_prompt(
            runner_target,
            runtime.extra_prompt_source,
            temp_files=runtime.temp_files,
            env=runner_env,
        )
    return runner_env, runner_target


def cleanup_runtime(runtime: LoopStepRuntime | None) -> None:
    """Discard a runtime's temporary files, tolerating one never prepared.

    A loop command's ``finally`` reaches here even when ``prepare_runtime``
    itself failed, so the absent runtime is this function's case to handle.
    """
    if runtime is not None:
        cleanup_temp_files(runtime.temp_files)


def report_interrupt(runtime: LoopStepRuntime | None) -> None:
    """Announce an interruption, logging it against a prepared runtime."""
    message = "\nInterrupted\n"
    print(message, end="")
    if runtime is not None:
        append_log(runtime.log_file, message)


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
        report_interrupt(runtime)
        raise SystemExit(130)
    finally:
        cleanup_runtime(runtime)
