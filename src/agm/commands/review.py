"""Review agent command."""

from __future__ import annotations

import os
from pathlib import Path

from agm.agent.prompt_source import PromptSourceOptions, resolve_prompt_source
from agm.agent.runner import (
    PreparedPromptRun,
    cleanup_temp_files,
    prepare_prompt_run,
    run_prepared_prompt,
)
from agm.commands.agent_io import (
    StreamCallback,
    default_agent_runner,
    exit_config_command_not_found,
    write_stderr,
    write_stdout,
)
from agm.commands.args import ReviewArgs
from agm.config.context import current_config_context
from agm.config.general import (
    ConfigCommandNotFound,
    ReviewConfig,
    load_review_config,
    resolve_default_prompt_file,
)
from agm.core import dry_run

DEFAULT_REVIEW_SCOPE = "changes on current branch"
DEFAULT_REVIEW_ASPECTS = "correctness, completeness, maintainability, adherence to AGENTS.md"

def _review_config(command_name: str | None, *, require_command: bool) -> ReviewConfig:
    context = current_config_context()
    try:
        return load_review_config(
            home=context.home,
            proj_dir=context.proj_dir,
            cwd=context.cwd,
            command_name=command_name,
            require_command=require_command,
        )
    except ConfigCommandNotFound as error:
        exit_config_command_not_found(error)


def _resolved_review_aspects(args: ReviewArgs, config: ReviewConfig) -> str:
    aspects = args.aspects or config.aspects or DEFAULT_REVIEW_ASPECTS
    extra_aspects = args.extra_aspects or config.extra_aspects
    if extra_aspects is None:
        return aspects
    return f"{aspects}, {extra_aspects}"


def prepare_review(
    args: ReviewArgs, *, temp_files: list[Path] | None = None
) -> PreparedPromptRun:
    owned_temp_files: list[Path] = [] if temp_files is None else temp_files
    context = current_config_context()
    config = _review_config(args.command_name, require_command=args.require_command_config)
    runner = args.runner or config.runner or default_agent_runner()
    scope = args.scope or config.scope or DEFAULT_REVIEW_SCOPE
    aspects = _resolved_review_aspects(args, config)
    env = dict(os.environ)
    env["REVIEW_SCOPE"] = scope
    env["REVIEW_ASPECTS"] = aspects
    default_prompt_file = resolve_default_prompt_file("review.md", home=context.home)
    prompt_source = resolve_prompt_source(
        PromptSourceOptions(
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            config_prompt=config.prompt,
            config_prompt_file=config.prompt_file,
            default_prompt_file=default_prompt_file,
        ),
        cwd=context.cwd,
    )
    assert prompt_source is not None
    extra_prompt_source = resolve_prompt_source(
        PromptSourceOptions(
            prompt=args.extra_prompt,
            prompt_file=args.extra_prompt_file,
            config_prompt=config.extra_prompt,
            config_prompt_file=config.extra_prompt_file,
        ),
        cwd=context.cwd,
    )
    return prepare_prompt_run(
        runner=runner,
        prompt_source=prompt_source,
        extra_prompt_source=extra_prompt_source,
        env=env,
        temp_files=owned_temp_files,
        kind="review runner",
    )


def review_once(
    args: ReviewArgs,
    *,
    stdout_callback: StreamCallback = write_stdout,
    stderr_callback: StreamCallback = write_stderr,
) -> str:
    temp_files: list[Path] = []
    try:
        prepared = prepare_review(args, temp_files=temp_files)
        if dry_run.enabled():
            dry_run.print_configuration("review")
        return run_prepared_prompt(
            prepared,
            stdout_callback=stdout_callback,
            stderr_callback=stderr_callback,
        )
    finally:
        cleanup_temp_files(temp_files)


def run(args: ReviewArgs) -> None:
    try:
        review_once(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
