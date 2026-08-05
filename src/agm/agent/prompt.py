"""Helpers for preparing prompt files before passing them to external tools."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NoReturn

from agm.core.fs import is_file
from agm.core.path import display_path
from agm.util.interp import InterpolationError, Segment, interp_segments, split_template


def prompt_source_label(source: str | Path | None) -> str:
    """Describe a prompt source for user-facing output."""
    if isinstance(source, Path):
        return display_path(source)
    return "inline prompt"


def require_prompt_file(path: Path, *, label: str = "prompt") -> None:
    """Exit with a CLI error when a prompt file does not exist."""
    if not is_file(path):
        print(f"Error: {label} file not found: {display_path(path)}", file=sys.stderr)
        raise SystemExit(1)


def _exit_interp_error(exc: InterpolationError, what: str) -> NoReturn:
    print(f"Error: cannot interpolate {what}: {exc}.", file=sys.stderr)
    raise SystemExit(1) from exc


def split_or_exit(template: str, *, what: str) -> list[Segment]:
    """Split a template, reporting malformed holes as CLI errors."""
    try:
        return split_template(template)
    except InterpolationError as exc:
        _exit_interp_error(exc, what)


def interp_or_exit(
    segments: Iterable[Segment],
    variables: Mapping[str, str],
    *,
    what: str,
) -> str:
    """Render split segments, reporting unavailable variables as CLI errors."""
    try:
        return interp_segments(segments, variables)
    except InterpolationError as exc:
        _exit_interp_error(exc, what)


def validate_prompt_template(
    source: str | Path,
    *,
    variables: Mapping[str, str],
    label: str | None = None,
) -> None:
    """Validate a prompt template's holes without rendering or writing it.

    Splits *source* (inline text, or a file path that must already exist)
    and checks every hole name against *variables*, reporting a malformed
    template or an unavailable hole the same way full interpolation would.
    The rendered text is discarded — callers use this to catch a broken
    template before doing side-effecting work such as invoking an agent.
    """
    content = source if isinstance(source, str) else source.read_text(encoding="utf-8")
    what = label if label is not None else prompt_source_label(source)
    interp_or_exit(split_or_exit(content, what=what), variables, what=what)


def expand_prompt_env_vars(
    content: str,
    *,
    env: Mapping[str, str],
    source: Path | None = None,
) -> str:
    """Expand named holes in a prompt, reporting prompt-specific CLI errors."""
    label = prompt_source_label(source)
    return interp_or_exit(split_or_exit(content, what=label), env, what=label)


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
