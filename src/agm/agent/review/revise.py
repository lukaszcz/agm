"""Revise pass logic."""

from __future__ import annotations

from pathlib import Path

from agm.agent.config import default_agent_runner
from agm.agent.io import StreamCallback, write_stderr, write_stdout
from agm.agent.prompt_source import PromptSourceOptions
from agm.agent.review.prompt_pass import prepare_prompt_pass
from agm.agent.runner import (
    PreparedPromptRun,
    cleanup_temp_files,
    run_prepared_prompt,
)
from agm.cli_support.args import ReviseArgs
from agm.config.command_config import load_command_config
from agm.config.context import current_config_context
from agm.config.general import (
    ConfigCommandNotFound,
    ReviseConfig,
    load_revise_config,
    resolve_default_prompt_file,
)
from agm.core import dry_run
from agm.core.env import clone_env
from agm.core.path import path_from_cli
from agm.parser import exit_with_usage_error


def _revise_config(command_name: str | None, *, require_command: bool) -> ReviseConfig:
    return load_command_config(load_revise_config, command_name, require_command=require_command)


def _exit_if_lone_revise_command_name(args: ReviseArgs) -> None:
    if args.command_name is not None:
        return
    context = current_config_context()
    if path_from_cli(args.review_file, cwd=context.cwd).exists():
        return
    try:
        load_revise_config(
            home=context.home,
            proj_dir=context.proj_dir,
            cwd=context.cwd,
            command_name=args.review_file,
        )
    except ConfigCommandNotFound:
        return
    exit_with_usage_error(
        ["revise"],
        f"error: revise command {args.review_file!r} was provided without REVIEW_FILE",
    )


def prepare_revise(args: ReviseArgs, *, temp_files: list[Path] | None = None) -> PreparedPromptRun:
    owned_temp_files: list[Path] = [] if temp_files is None else temp_files
    _exit_if_lone_revise_command_name(args)
    context = current_config_context()
    config = _revise_config(args.command_name, require_command=args.require_command_config)
    runner = args.runner or config.runner or default_agent_runner()
    env = clone_env()
    review_file = path_from_cli(args.review_file, cwd=context.cwd)
    env["REVIEW_FILE"] = str(review_file)
    default_prompt_file = resolve_default_prompt_file("revise.md", home=context.home)
    return prepare_prompt_pass(
        runner=runner,
        primary=PromptSourceOptions(
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            config_prompt=config.prompt,
            config_prompt_file=config.prompt_file,
            default_prompt_file=default_prompt_file,
        ),
        extra=PromptSourceOptions(
            prompt=args.extra_prompt,
            prompt_file=args.extra_prompt_file,
            config_prompt=config.extra_prompt,
            config_prompt_file=config.extra_prompt_file,
        ),
        env=env,
        temp_files=owned_temp_files,
        kind="revise runner",
        cwd=context.cwd,
    )


def revise_once(
    args: ReviseArgs,
    *,
    stdout_callback: StreamCallback = write_stdout,
    stderr_callback: StreamCallback = write_stderr,
) -> str:
    temp_files: list[Path] = []
    try:
        prepared = prepare_revise(args, temp_files=temp_files)
        if dry_run.enabled():
            dry_run.print_configuration("revise")
        return run_prepared_prompt(
            prepared,
            stdout_callback=stdout_callback,
            stderr_callback=stderr_callback,
        )
    finally:
        cleanup_temp_files(temp_files)
