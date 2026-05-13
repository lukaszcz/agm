"""Refine agent command."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from agm.agent.output import step_header_text
from agm.agent.response import last_response_line
from agm.agent.runner import cleanup_temp_files
from agm.commands.agent_io import exit_config_command_not_found, write_stderr, write_stdout
from agm.commands.args import RefineArgs, ReviewArgs, ReviseArgs
from agm.commands.review import review_once
from agm.commands.revise import revise_once
from agm.config.context import current_config_context
from agm.config.general import ConfigCommandNotFound, RefineConfig, load_refine_config
from agm.core.log import append_log, prepare_log_file, resolve_log_file

DEFAULT_MAX_STEPS = 20


def _refine_config(command_name: str | None) -> RefineConfig:
    context = current_config_context()
    try:
        return load_refine_config(
            home=context.home,
            proj_dir=context.proj_dir,
            cwd=context.cwd,
            command_name=command_name,
        )
    except ConfigCommandNotFound as error:
        exit_config_command_not_found(error)


def _write_review_file(output: str, *, temp_files: list[Path]) -> Path:
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as handle:
        handle.write(output)
        path = Path(handle.name)
    temp_files.append(path)
    return path


def _unlink_temp_file(path: Path, *, temp_files: list[Path]) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    try:
        temp_files.remove(path)
    except ValueError:
        pass


def _review_args_from_refine(args: RefineArgs, config: RefineConfig) -> ReviewArgs:
    runner = args.reviewer or args.runner or config.reviewer or config.runner
    return ReviewArgs(
        runner=runner,
        scope=args.scope or config.scope,
        aspects=args.aspects or config.aspects,
        extra_aspects=None,
        prompt=args.review_prompt or config.review_prompt,
        prompt_file=args.review_prompt_file or config.review_prompt_file,
        extra_prompt=args.extra_review_prompt or config.extra_review_prompt,
        extra_prompt_file=args.extra_review_prompt_file or config.extra_review_prompt_file,
        command_name=args.command_name,
        require_command_config=False,
        review_file="auto" if args.save_review or config.save_review else None,
        no_review_file=not (args.save_review or config.save_review),
    )


def _revise_args_from_refine(
    args: RefineArgs,
    config: RefineConfig,
    review_file: Path,
) -> ReviseArgs:
    runner = args.reviser or args.runner or config.reviser or config.runner
    return ReviseArgs(
        review_file=str(review_file),
        runner=runner,
        prompt=args.revise_prompt or config.revise_prompt,
        prompt_file=args.revise_prompt_file or config.revise_prompt_file,
        extra_prompt=args.extra_revise_prompt or config.extra_revise_prompt,
        extra_prompt_file=args.extra_revise_prompt_file or config.extra_revise_prompt_file,
        command_name=args.command_name,
        require_command_config=False,
    )


def refine(args: RefineArgs) -> None:
    config = _refine_config(args.command_name)
    max_steps = args.max_steps or config.max_steps or DEFAULT_MAX_STEPS
    log_file = resolve_log_file(
        command_name="refine",
        no_log=args.no_log,
        log_file=args.log_file,
    )
    prepare_log_file(log_file, explicit=args.log_file is not None)

    def stdout_callback(chunk: str) -> None:
        append_log(log_file, chunk)
        write_stdout(chunk)

    def stderr_callback(chunk: str) -> None:
        append_log(log_file, chunk)
        write_stderr(chunk)

    temp_files: list[Path] = []
    try:
        step = 0
        review_file: Path | None = None
        while step < max_steps:
            step += 1
            header = step_header_text(step)
            append_log(log_file, header)
            write_stdout(header)
            if review_file is None:
                review_args = _review_args_from_refine(args, config)
                review_output = review_once(
                    review_args,
                    stdout_callback=stdout_callback,
                    stderr_callback=stderr_callback,
                )
                review_file = _write_review_file(review_output, temp_files=temp_files)
            revise_output = revise_once(
                _revise_args_from_refine(args, config, review_file),
                stdout_callback=stdout_callback,
                stderr_callback=stderr_callback,
            )
            status = last_response_line(revise_output)
            if status == "COMPLETE":
                return
            if status == "CONTINUE":
                _unlink_temp_file(review_file, temp_files=temp_files)
                review_file = None
    finally:
        cleanup_temp_files(temp_files)


def run(args: RefineArgs) -> None:
    try:
        refine(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
