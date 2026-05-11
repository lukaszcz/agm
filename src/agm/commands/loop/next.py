"""agm loop next."""

from __future__ import annotations

import sys
from pathlib import Path

from agm.commands.args import LoopNextArgs
from agm.core import dry_run

from .common import (
    append_extra_prompt,
    cleanup_temp_files,
    command_with_prompt_target,
    loop_env,
    prepare_select_invocation,
    resolve_extra_selector_prompt_source,
    resolved_timeout,
    run_command,
    tasks_dir,
    use_selector_mode,
)


def _dry_run_prompt_text(source_file: Path, effective_file: Path) -> str:
    if source_file == effective_file:
        return str(source_file)
    return f"{source_file} -> {effective_file} (preprocessed)"


def _print_dry_run_prompt(label: str, prompt_text: str) -> None:
    print(f"dry-run: prompt [{label}]: {prompt_text}")


def run(args: LoopNextArgs) -> None:
    temp_files: list[Path] = []
    resolved_tasks_dir = tasks_dir(args)
    env = loop_env(resolved_tasks_dir)
    timeout = resolved_timeout(args)

    if not use_selector_mode(args):
        print(
            "Error: agm loop next requires selector mode. "
            "Remove --no-selector to enable it.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        invocation = prepare_select_invocation(args, temp_files=temp_files, env=env)

        extra_selector_prompt_source = resolve_extra_selector_prompt_source(args)
        if extra_selector_prompt_source is not None:
            invocation.effective_prompt_file = append_extra_prompt(
                invocation.effective_prompt_file,
                extra_selector_prompt_source,
                temp_files=temp_files,
                env=env,
            )

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
                "selector",
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

        print(
            run_command(
                invocation.command,
                invocation.effective_prompt_file,
                env=env,
                idle_timeout=timeout,
            ),
            end="",
        )
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
    finally:
        cleanup_temp_files(temp_files)
