"""Shared prompt preparation helper for review and revise passes."""

from __future__ import annotations

from pathlib import Path

from agm.agent.prompt_source import PromptSourceOptions, resolve_prompt_source
from agm.agent.runner import PreparedPromptRun, prepare_prompt_run


def prepare_prompt_pass(
    *,
    runner: str,
    primary: PromptSourceOptions,
    extra: PromptSourceOptions,
    env: dict[str, str],
    temp_files: list[Path],
    kind: str,
    cwd: Path,
) -> PreparedPromptRun:
    prompt_source = resolve_prompt_source(primary, cwd=cwd)
    assert prompt_source is not None
    extra_prompt_source = resolve_prompt_source(extra, cwd=cwd)
    return prepare_prompt_run(
        runner=runner,
        prompt_source=prompt_source,
        extra_prompt_source=extra_prompt_source,
        env=env,
        temp_files=temp_files,
        kind=kind,
    )
