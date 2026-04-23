"""Shared helpers for loop commands."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from agm.commands.args import LoopArgs, LoopProgressArgs
from agm.config.general import LoopConfig, load_loop_config
from agm.core.env import agm_installation_prefix
from agm.core.fs import is_file
from agm.core.process import run_capture

LoopCommandArgs = LoopArgs | LoopProgressArgs


def prompt_dir_candidates() -> list[Path]:
    candidates: list[Path] = []

    install_prefix = agm_installation_prefix()
    if install_prefix is not None:
        candidates.append(install_prefix / ".agm" / "prompts")

    home = Path(os.environ["HOME"])
    candidates.append(home / ".agm" / "prompts")

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def prompt_file(filename: str) -> Path:
    candidates = [prompt_dir / filename for prompt_dir in prompt_dir_candidates()]
    for candidate in candidates:
        if is_file(candidate):
            return candidate
    return candidates[-1]


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
    return f"Selected task: {task_file}\n"


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


def loop_env(tasks_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["TASKS_DIR"] = str(tasks_dir)
    return env


def run_command(
    command: list[str],
    target: Path,
    *,
    env: dict[str, str],
    stdout_callback: Callable[[str], None] | None = None,
    stderr_callback: Callable[[str], None] | None = None,
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
    )
    if ordered_output:
        return "".join(ordered_output)
    output = stdout
    if stderr:
        output += stderr
    return output


def selector_result(output: str, *, tasks_dir: Path) -> Path | None | str:
    selected = output.strip()
    if not selected:
        return ""
    if "".join(selected.split()) == "COMPLETE":
        return None

    task_path = Path(selected)
    if not task_path.is_absolute():
        task_path = tasks_dir / task_path
    if not is_file(task_path):
        return selected
    return task_path


def cleanup_temp_files(temp_files: list[Path]) -> None:
    for temp_file in temp_files:
        try:
            temp_file.unlink()
        except FileNotFoundError:
            pass
