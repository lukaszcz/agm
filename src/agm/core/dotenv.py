"""Dotenv file editing helpers."""

from __future__ import annotations

import re
from pathlib import Path

from agm.core.fs import exists, mkdir, read_text, write_text


def set_dotenv_value(path: Path, key: str, value: str) -> None:
    """Set *key* to *value* in a dotenv file."""

    key_pattern = re.compile(rf"^(?:export\s+)?{re.escape(key)}\s*=")
    assignment = f"{key}={value}\n"
    lines = read_text(path).splitlines(keepends=True) if exists(path) else []
    updated_lines: list[str] = []
    replaced = False

    for line in lines:
        if key_pattern.match(line):
            if not replaced:
                updated_lines.append(assignment)
                replaced = True
            continue
        updated_lines.append(line)

    if not replaced:
        if updated_lines and not updated_lines[-1].endswith("\n"):
            updated_lines[-1] = f"{updated_lines[-1]}\n"
        updated_lines.append(assignment)

    mkdir(path.parent, parents=True, exist_ok=True)
    write_text(path, "".join(updated_lines))
