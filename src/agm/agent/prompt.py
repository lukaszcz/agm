"""Helpers for preparing prompt files before passing them to external tools."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile

from agm.core.path import display_path
from agm.util.interp import InterpolationError, interp


def expand_prompt_env_vars(
    content: str,
    *,
    env: Mapping[str, str],
    source: Path | None = None,
) -> str:
    """Expand named holes in a prompt, reporting prompt-specific CLI errors."""
    try:
        return interp(content, env)
    except InterpolationError as exc:
        prompt_source = "inline prompt" if source is None else display_path(source)
        print(f"Error: cannot interpolate {prompt_source}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def preprocess_prompt_file(
    prompt_file: Path,
    *,
    temp_files: list[Path],
    env: Mapping[str, str],
) -> Path:
    original = prompt_file.read_text(encoding="utf-8")
    expanded = expand_prompt_env_vars(original, env=env, source=prompt_file)
    if expanded == original:
        return prompt_file
    with NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(expanded)
        path = Path(handle.name)
    temp_files.append(path)
    return path


def dry_run_prompt_text(source_file: Path, effective_file: Path) -> str:
    if source_file == effective_file:
        return display_path(source_file)
    return f"{display_path(source_file)} -> {display_path(effective_file)} (preprocessed)"
