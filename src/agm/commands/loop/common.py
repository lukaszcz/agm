"""Shared helpers for loop commands."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from agm.commands.args import LoopArgs, LoopNextArgs
from agm.config.general import LoopConfig, load_loop_config, resolve_agm_path
from agm.core.fs import is_file
from agm.core.process import run_capture
from agm.core.prompt import expand_prompt_env_vars, preprocess_prompt_file

LoopCommandArgs = LoopArgs | LoopNextArgs


@dataclass(slots=True)
class ResolvedPrompt:
    """Resolved prompt source: either inline text or a file path."""

    source: str | Path
    effective_file: Path


@dataclass(slots=True)
class PreparedSelectInvocation:
    source_prompt_file: Path
    effective_prompt_file: Path
    command: list[str]
    command_kind: str
    runner_command: list[str]
    selector_command: list[str] | None


def prompt_file(filename: str) -> Path:
    return resolve_agm_path(
        home=Path(os.environ["HOME"]),
        relative_path=Path("prompts") / filename,
    )


def configured_loop_settings(command_name: str | None) -> LoopConfig:
    return load_loop_config(
        home=Path(os.environ["HOME"]),
        proj_dir=Path(os.environ["PROJ_DIR"]) if os.environ.get("PROJ_DIR") else None,
        cwd=Path.cwd(),
        command_name=command_name,
    )


def step_header_text(step: int) -> str:
    return (
        "\n"
        "-------------------------------------------------------------\n"
        f"                        Step {step}\n"
        "-------------------------------------------------------------\n"
        "\n"
    )


def selected_task_text(task_file: Path) -> str:
    return f"Selected task: {task_file}\n\n"


def split_command(command: str, *, kind: str) -> list[str]:
    split = shlex.split(command)
    if split:
        return split
    print(f"Error: loop {kind} command is empty.", file=sys.stderr)
    raise SystemExit(1)


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
    if selected is None:
        return Path.cwd() / ".agent-files" / "tasks"

    resolved_tasks_dir = Path(selected)
    if resolved_tasks_dir.is_absolute():
        return resolved_tasks_dir
    return Path.cwd() / resolved_tasks_dir


def progress_file(args: LoopCommandArgs) -> Path:
    return tasks_dir(args) / "PROGRESS.md"


def validate_command(command: list[str], *, kind: str) -> None:
    if shutil.which(command[0]) is None:
        print(
            f"Error: {kind} command {command[0]} is not installed or not in PATH.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def command_with_prompt_target(command: list[str], target: Path) -> list[str]:
    prompt_path = str(target)
    placeholders = ("%%", "%{PROMPT_FILE}")
    replaced_command: list[str] = []
    replaced = False

    for arg in command:
        updated = arg
        for placeholder in placeholders:
            if placeholder in updated:
                updated = updated.replace(placeholder, prompt_path)
                replaced = True
        replaced_command.append(updated)

    if replaced:
        return replaced_command
    return [*command, f"@{target}"]


def resolve_prompt_source(args: LoopCommandArgs) -> str | Path | None:
    """Resolve the prompt source from CLI args and config.

    Returns the prompt text (str), prompt file path (Path), or None when
    neither --prompt nor --prompt-file is specified.
    """
    configured = configured_loop_settings(args.command_name)
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return Path(args.prompt_file)
    if configured.prompt is not None:
        return configured.prompt
    if configured.prompt_file is not None:
        resolved = Path(configured.prompt_file)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        return resolved
    return None


def resolve_selector_prompt_source(args: LoopCommandArgs) -> str | Path | None:
    """Resolve the selector prompt source from CLI args and config.

    Returns the prompt text (str), prompt file path (Path), or None when
    neither --selector-prompt nor --selector-prompt-file is specified.
    """
    configured = configured_loop_settings(args.command_name)
    if args.selector_prompt is not None:
        return args.selector_prompt
    if args.selector_prompt_file is not None:
        return Path(args.selector_prompt_file)
    if configured.selector_prompt is not None:
        return configured.selector_prompt
    if configured.selector_prompt_file is not None:
        resolved = Path(configured.selector_prompt_file)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        return resolved
    return None


def prepare_prompt_from_source(
    source: str | Path,
    *,
    temp_files: list[Path],
    env: dict[str, str],
) -> ResolvedPrompt:
    """Create a preprocessed prompt file from inline text or a file path.

    When ``source`` is a string it is written to a temporary file and then
    preprocessed.  When ``source`` is a ``Path`` the file is preprocessed
    in place (same as ``loop.md`` handling).
    """
    if isinstance(source, str):
        expanded = expand_prompt_env_vars(source, env=env)
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as handle:
            handle.write(expanded)
            temp_path = Path(handle.name)
        temp_files.append(temp_path)
        return ResolvedPrompt(source=source, effective_file=temp_path)

    source_path = source
    if not is_file(source_path):
        print(
            f"Error: prompt file not found: {source_path}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    effective = preprocess_prompt_file(source_path, temp_files=temp_files, env=env)
    return ResolvedPrompt(source=source_path, effective_file=effective)


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

    selector_prompt_source = resolve_selector_prompt_source(args)
    if selector_prompt_source is not None:
        resolved = prepare_prompt_from_source(
            selector_prompt_source, temp_files=temp_files, env=env
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


def run_command(
    command: list[str],
    target: Path,
    *,
    env: dict[str, str],
    stdout_callback: Callable[[str], None] | None = None,
    stderr_callback: Callable[[str], None] | None = None,
    idle_timeout: float | None = None,
) -> str:
    ordered_output: list[str] = []

    def handle_stdout(chunk: str) -> None:
        ordered_output.append(chunk)
        if stdout_callback is not None:
            stdout_callback(chunk)

    def handle_stderr(chunk: str) -> None:
        ordered_output.append(chunk)
        if stderr_callback is not None:
            stderr_callback(chunk)

    _, stdout, stderr = run_capture(
        command_with_prompt_target(command, target),
        env=env,
        stdout_callback=handle_stdout,
        stderr_callback=handle_stderr,
        isolate_process_group=True,
        idle_timeout=idle_timeout,
    )
    if ordered_output:
        return "".join(ordered_output)
    output = stdout
    if stderr:
        output += stderr
    return output


def last_output_line(output: str) -> str:
    lines = output.splitlines()
    if not lines:
        return output.strip()
    return lines[-1].strip()


def is_complete_output(output: str) -> bool:
    return "".join(last_output_line(output).split()) == "COMPLETE"


def selector_result(output: str, *, tasks_dir: Path) -> Path | None | str:
    selected = last_output_line(output)
    if not selected:
        return ""
    if is_complete_output(output):
        return None

    task_path = Path(selected)
    if task_path.is_absolute():
        if is_file(task_path):
            return task_path
        return selected

    resolved_task_path = Path.cwd() / task_path
    if is_file(resolved_task_path):
        return resolved_task_path
    tasks_dir_task_path = tasks_dir / task_path
    if is_file(tasks_dir_task_path):
        return tasks_dir_task_path
    return selected


def cleanup_temp_files(temp_files: list[Path]) -> None:
    for temp_file in temp_files:
        try:
            temp_file.unlink()
        except FileNotFoundError:
            pass
