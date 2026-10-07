"""Shared prompt preparation helper for review and revise passes."""

from __future__ import annotations

from pathlib import Path

from agm.agent.prompt_source import PromptSourceOptions, resolve_prompt_source
from agm.agent.runner import PreparedPromptRun, prepare_prompt_run


def prepare_prompt_pass(
    *,
    runner: str,
    primary: PromptSourceOptions,
    default_prompt_file: Path,
    extra: PromptSourceOptions,
    env: dict[str, str],
    temp_files: list[Path],
    kind: str,
    cwd: Path,
) -> PreparedPromptRun:
    prompt_source = resolve_prompt_source(primary, cwd=cwd)
    extra_prompt_source = resolve_prompt_source(extra, cwd=cwd)
    return prepare_prompt_run(
        runner=runner,
        prompt_source=default_prompt_file if prompt_source is None else prompt_source,
        extra_prompt_source=extra_prompt_source,
        env=env,
        temp_files=temp_files,
        kind=kind,
    )
