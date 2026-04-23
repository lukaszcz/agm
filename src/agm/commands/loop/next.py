"""agm loop next."""

from __future__ import annotations

import sys
from pathlib import Path

from agm.commands.args import LoopProgressArgs
from agm.core import dry_run
from agm.core.fs import is_file
from agm.core.prompt import preprocess_prompt_file

from .common import (
    cleanup_temp_files,
    command_with_prompt_target,
    loop_env,
    prompt_file,
    run_command,
    runner_command,
    selector_command,
    tasks_dir,
    validate_command,
)


def _dry_run_prompt_text(source_file: Path, effective_file: Path) -> str:
    if source_file == effective_file:
        return str(source_file)
    return f"{source_file} -> {effective_file} (preprocessed)"


def _print_dry_run_prompt(label: str, prompt_text: str) -> None:
    print(f"dry-run: prompt [{label}]: {prompt_text}")


def run(args: LoopProgressArgs) -> None:
    temp_files: list[Path] = []
    resolved_tasks_dir = tasks_dir(args)
    env = loop_env(resolved_tasks_dir)
    resolved_runner_command = runner_command(args)
    resolved_selector_command = selector_command(args)
    resolved_command = resolved_selector_command
    command_kind = "selector"
    if resolved_command is None:
        resolved_command = resolved_runner_command
        command_kind = "runner"
    validate_command(resolved_command, kind=command_kind)

    try:
        source_prompt_file = prompt_file("update_progress.md")
        if not is_file(source_prompt_file):
            print(f"Error: prompt file not found: {source_prompt_file}", file=sys.stderr)
            raise SystemExit(1)
        resolved_prompt_file = preprocess_prompt_file(
            source_prompt_file, temp_files=temp_files, env=env
        )

        if dry_run.enabled():
            dry_run.print_configuration("loop-next")
            dry_run.print_detail("tasks dir", str(resolved_tasks_dir))
            dry_run.print_detail("runner command", dry_run.format_command(resolved_runner_command))
            selector_text = (
                dry_run.format_command(resolved_selector_command)
                if resolved_selector_command is not None
                else "disabled"
            )
            dry_run.print_detail("selector command", selector_text)
            dry_run.print_detail("execution command", dry_run.format_command(resolved_command))
            _print_dry_run_prompt(
                "progress",
                _dry_run_prompt_text(source_prompt_file, resolved_prompt_file),
            )
            dry_run.print_labeled_command(
                command_kind,
                command_with_prompt_target(resolved_command, resolved_prompt_file),
            )
            return

        print(run_command(resolved_command, resolved_prompt_file, env=env), end="")
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
    finally:
        cleanup_temp_files(temp_files)
