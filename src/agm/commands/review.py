"""Review, revise, and refine agent commands."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from agm.commands.args import RefineArgs, ReviewArgs, ReviseArgs
from agm.config.general import (
    RefineConfig,
    ReviewConfig,
    ReviseConfig,
    load_loop_config,
    load_refine_config,
    load_review_config,
    load_revise_config,
    resolve_agm_path,
)
from agm.core import dry_run
from agm.core.agent import (
    append_extra_prompt,
    cleanup_temp_files,
    command_with_prompt_target,
    prepare_prompt_from_source,
    run_prompt_command,
    split_command,
    validate_command,
)
from agm.core.fs import is_file
from agm.core.response import last_response_line

DEFAULT_REVIEW_SCOPE = "changes on current branch"
DEFAULT_REVIEW_ASPECTS = "correctness, completeness, maintainability, adherence to AGENTS.md"
DEFAULT_MAX_STEPS = 20


@dataclass(slots=True)
class AgentPromptRun:
    command: list[str]
    source_file: Path
    effective_file: Path
    env: dict[str, str]
    temp_files: list[Path]


def _project_dir() -> Path | None:
    value = os.environ.get("PROJ_DIR")
    return Path(value) if value else None


def _config_home() -> Path:
    return Path(os.environ["HOME"])


def _default_runner() -> str:
    loop_config = load_loop_config(home=_config_home(), proj_dir=_project_dir(), cwd=Path.cwd())
    return loop_config.runner if loop_config.runner is not None else "claude -p"


def _prompt_file(filename: str) -> Path:
    return resolve_agm_path(
        home=_config_home(),
        relative_path=Path("prompts") / filename,
    )


def _path_from_cli(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _prompt_source(
    *,
    prompt: str | None,
    prompt_file: str | None,
    config_prompt: str | None,
    config_prompt_file: str | None,
    default_prompt_file: Path,
) -> str | Path:
    if prompt is not None:
        return prompt
    if prompt_file is not None:
        return _path_from_cli(prompt_file)
    if config_prompt is not None:
        return config_prompt
    if config_prompt_file is not None:
        return Path(config_prompt_file)
    return default_prompt_file


def _extra_prompt_source(
    *,
    extra_prompt: str | None,
    extra_prompt_file: str | None,
    config_extra_prompt: str | None,
    config_extra_prompt_file: str | None,
) -> str | Path | None:
    if extra_prompt is not None:
        return extra_prompt
    if extra_prompt_file is not None:
        return _path_from_cli(extra_prompt_file)
    if config_extra_prompt is not None:
        return config_extra_prompt
    if config_extra_prompt_file is not None:
        return Path(config_extra_prompt_file)
    return None


def _write_stdout(chunk: str) -> None:
    if not chunk:
        return
    sys.stdout.write(chunk)
    sys.stdout.flush()


def _write_stderr(chunk: str) -> None:
    if not chunk:
        return
    sys.stderr.write(chunk)
    sys.stderr.flush()


def _prepare_agent_prompt_run(
    *,
    runner: str,
    prompt_source: str | Path,
    default_prompt_file: Path,
    extra_prompt_source: str | Path | None,
    env: dict[str, str],
    temp_files: list[Path],
    kind: str,
) -> AgentPromptRun:
    command = split_command(runner, kind=kind)
    validate_command(command, kind=kind)
    if isinstance(prompt_source, Path):
        source_file = prompt_source
    else:
        source_file = default_prompt_file
    resolved = prepare_prompt_from_source(prompt_source, temp_files=temp_files, env=env)
    effective_file = resolved.effective_file
    if extra_prompt_source is not None:
        effective_file = append_extra_prompt(
            effective_file,
            extra_prompt_source,
            temp_files=temp_files,
            env=env,
        )
    return AgentPromptRun(
        command=command,
        source_file=source_file,
        effective_file=effective_file,
        env=env,
        temp_files=temp_files,
    )


def _run_prepared(prepared: AgentPromptRun) -> str:
    if dry_run.enabled():
        dry_run.print_labeled_command(
            "agent",
            command_with_prompt_target(prepared.command, prepared.effective_file),
        )
        return ""
    return run_prompt_command(
        prepared.command,
        prepared.effective_file,
        env=prepared.env,
        stdout_callback=_write_stdout,
        stderr_callback=_write_stderr,
    )


def _review_config() -> ReviewConfig:
    return load_review_config(home=_config_home(), proj_dir=_project_dir(), cwd=Path.cwd())


def _revise_config() -> ReviseConfig:
    return load_revise_config(home=_config_home(), proj_dir=_project_dir(), cwd=Path.cwd())


def _refine_config() -> RefineConfig:
    return load_refine_config(home=_config_home(), proj_dir=_project_dir(), cwd=Path.cwd())


def _resolved_review_aspects(args: ReviewArgs, config: ReviewConfig) -> str:
    aspects = args.aspects or config.aspects or DEFAULT_REVIEW_ASPECTS
    extra_aspects = args.extra_aspects or config.extra_aspects
    if extra_aspects is None:
        return aspects
    return f"{aspects}, {extra_aspects}"


def prepare_review(args: ReviewArgs, *, temp_files: list[Path] | None = None) -> AgentPromptRun:
    owned_temp_files: list[Path] = [] if temp_files is None else temp_files
    config = _review_config()
    runner = args.runner or config.runner or _default_runner()
    scope = args.scope or config.scope or DEFAULT_REVIEW_SCOPE
    aspects = _resolved_review_aspects(args, config)
    env = dict(os.environ)
    env["REVIEW_SCOPE"] = scope
    env["REVIEW_ASPECTS"] = aspects
    default_prompt_file = _prompt_file("review.md")
    if not is_file(default_prompt_file) and args.prompt is None and args.prompt_file is None:
        print(f"Error: prompt file not found: {default_prompt_file}", file=sys.stderr)
        raise SystemExit(1)
    return _prepare_agent_prompt_run(
        runner=runner,
        prompt_source=_prompt_source(
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            config_prompt=config.prompt,
            config_prompt_file=config.prompt_file,
            default_prompt_file=default_prompt_file,
        ),
        default_prompt_file=default_prompt_file,
        extra_prompt_source=_extra_prompt_source(
            extra_prompt=args.extra_prompt,
            extra_prompt_file=args.extra_prompt_file,
            config_extra_prompt=config.extra_prompt,
            config_extra_prompt_file=config.extra_prompt_file,
        ),
        env=env,
        temp_files=owned_temp_files,
        kind="review runner",
    )


def prepare_revise(args: ReviseArgs, *, temp_files: list[Path] | None = None) -> AgentPromptRun:
    owned_temp_files: list[Path] = [] if temp_files is None else temp_files
    config = _revise_config()
    runner = args.runner or config.runner or _default_runner()
    env = dict(os.environ)
    review_file = _path_from_cli(args.review_file)
    env["REVIEW_FILE"] = str(review_file)
    default_prompt_file = _prompt_file("revise.md")
    if not is_file(default_prompt_file) and args.prompt is None and args.prompt_file is None:
        print(f"Error: prompt file not found: {default_prompt_file}", file=sys.stderr)
        raise SystemExit(1)
    return _prepare_agent_prompt_run(
        runner=runner,
        prompt_source=_prompt_source(
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            config_prompt=config.prompt,
            config_prompt_file=config.prompt_file,
            default_prompt_file=default_prompt_file,
        ),
        default_prompt_file=default_prompt_file,
        extra_prompt_source=_extra_prompt_source(
            extra_prompt=args.extra_prompt,
            extra_prompt_file=args.extra_prompt_file,
            config_extra_prompt=config.extra_prompt,
            config_extra_prompt_file=config.extra_prompt_file,
        ),
        env=env,
        temp_files=owned_temp_files,
        kind="revise runner",
    )


def review_once(args: ReviewArgs) -> str:
    temp_files: list[Path] = []
    try:
        prepared = prepare_review(args, temp_files=temp_files)
        if dry_run.enabled():
            dry_run.print_configuration("review")
        return _run_prepared(prepared)
    finally:
        cleanup_temp_files(temp_files)


def revise_once(args: ReviseArgs) -> str:
    temp_files: list[Path] = []
    try:
        prepared = prepare_revise(args, temp_files=temp_files)
        if dry_run.enabled():
            dry_run.print_configuration("revise")
        return _run_prepared(prepared)
    finally:
        cleanup_temp_files(temp_files)


def _write_review_file(output: str, *, temp_files: list[Path], env: dict[str, str]) -> Path:
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".md") as handle:
        handle.write(output)
        path = Path(handle.name)
    temp_files.append(path)
    return path


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
    )


def refine(args: RefineArgs) -> None:
    config = _refine_config()
    max_steps = args.max_steps or config.max_steps or DEFAULT_MAX_STEPS
    temp_files: list[Path] = []
    try:
        step = 0
        review_file: Path | None = None
        while step < max_steps:
            if review_file is None:
                review_args = _review_args_from_refine(args, config)
                review_output = review_once(review_args)
                review_file = _write_review_file(
                    review_output,
                    temp_files=temp_files,
                    env=dict(os.environ),
                )
            step += 1
            revise_output = revise_once(_revise_args_from_refine(args, config, review_file))
            status = last_response_line(revise_output)
            if status == "COMPLETE":
                return
            if status == "CONTINUE":
                review_file = None
    finally:
        cleanup_temp_files(temp_files)


def run_review(args: ReviewArgs) -> None:
    try:
        review_once(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)


def run_revise(args: ReviseArgs) -> None:
    try:
        revise_once(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)


def run_refine(args: RefineArgs) -> None:
    try:
        refine(args)
    except KeyboardInterrupt:
        print("\nInterrupted")
        raise SystemExit(130)
