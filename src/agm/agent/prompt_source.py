"""Prompt source precedence for prompt-driven agent commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptSourceOptions:
    """Candidate prompt sources in decreasing precedence."""

    prompt: str | None
    prompt_file: str | None
    config_prompt: str | None
    config_prompt_file: str | None
    default_prompt_file: Path | None = None


def path_from_cli(value: str, *, cwd: Path) -> Path:
    """Resolve a CLI path value relative to *cwd*."""

    path = Path(value)
    if path.is_absolute():
        return path
    return cwd / path


def resolve_prompt_source(
    options: PromptSourceOptions,
    *,
    cwd: Path,
) -> str | Path | None:
    """Resolve the effective prompt source from CLI, config, and default values."""

    if options.prompt is not None:
        return options.prompt
    if options.prompt_file is not None:
        return path_from_cli(options.prompt_file, cwd=cwd)
    if options.config_prompt is not None:
        return options.config_prompt
    if options.config_prompt_file is not None:
        return Path(options.config_prompt_file)
    return options.default_prompt_file
