"""Helpers for preparing prompt files before passing them to external tools."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile

_PROMPT_ENV_VAR_PATTERN = re.compile(
    r"\$(?P<simple>[A-Za-z_][A-Za-z0-9_]*)|\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}"
)


def expand_prompt_env_vars(content: str, *, env: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("simple") or match.group("braced")
        assert name is not None
        value = env.get(name)
        if value is None:
            return match.group(0)
        return value

    return _PROMPT_ENV_VAR_PATTERN.sub(replace, content)


def preprocess_prompt_file(
    prompt_file: Path,
    *,
    temp_files: list[Path],
    env: Mapping[str, str],
) -> Path:
    original = prompt_file.read_text(encoding="utf-8")
    expanded = expand_prompt_env_vars(original, env=env)
    if expanded == original:
        return prompt_file
    with NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(expanded)
        path = Path(handle.name)
    temp_files.append(path)
    return path
