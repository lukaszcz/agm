"""agm loop next."""

from __future__ import annotations

from pathlib import Path

from agm.commands.args import LoopProgressArgs
from agm.core import dry_run

from .common import (
    cleanup_temp_files,
    command_with_prompt_target,
    loop_env,
    prepare_progress_invocation,
    run_command,
    tasks_dir,
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

    try:
        invocation = prepare_progress_invocation(args, temp_files=temp_files, env=env)

        if dry_run.enabled():
            dry_run.print_configuration("loop-next")
            dry_run.print_detail("tasks dir", str(resolved_tasks_dir))
            dry_run.print_detail(
                "runner command",
                dry_run.format_command(invocation.runner_command),
            )
            selector_text = (
                dry_run.format_command(invocation.selector_command)
                if invocation.selector_command is not None
                else "disabled"
            )
            dry_run.print_detail("selector command", selector_text)
            dry_run.print_detail("execution command", dry_run.format_command(invocation.command))
            _print_dry_run_prompt(
                "progress",
                _dry_run_prompt_text(
                    invocation.source_prompt_file,
                    invocation.effective_prompt_file,
                ),
            )
            dry_run.print_labeled_command(
                invocation.command_kind,
                command_with_prompt_target(invocation.command, invocation.effective_prompt_file),
            )
            return

        print(run_command(invocation.command, invocation.effective_prompt_file, env=env), end="")
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
    finally:
        cleanup_temp_files(temp_files)
