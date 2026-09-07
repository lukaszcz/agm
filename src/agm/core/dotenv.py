"""Dotenv file editing helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from io import StringIO
from pathlib import Path

from dotenv.parser import parse_stream

from agm.core.fs import exists, mkdir, read_text, write_text_atomic

_QUOTED_VALUE_REQUIRED = re.compile(r"[\r\n]|^\s|\s$|^['\"]|\s+#")


def _format_value(value: str) -> str:
    if _QUOTED_VALUE_REQUIRED.search(value) is None:
        return value
    escaped = (
        value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "\\r").replace("\n", "\\n")
    )
    return f'"{escaped}"'


def _format_assignment(key: str, value: str) -> str:
    return f"{key}={_format_value(value)}\n"


def _leading_line_breaks(content: str) -> str:
    leading = content[: len(content) - len(content.lstrip())]
    last_break = max(leading.rfind("\r"), leading.rfind("\n"))
    return leading[: last_break + 1]


def write_dotenv_values(path: Path, values: Mapping[str, str]) -> None:
    """Replace *path* atomically with sorted dotenv assignments."""

    content = "".join(_format_assignment(key, value) for key, value in sorted(values.items()))
    mkdir(path.parent, parents=True, exist_ok=True)
    write_text_atomic(path, content)


def set_dotenv_values(path: Path, values: Mapping[str, str]) -> None:
    """Atomically upsert *values* in a dotenv file, preserving unrelated entries."""

    remaining = dict(values)
    updated_parts: list[str] = []
    content = read_text(path) if exists(path) else ""
    for binding in parse_stream(StringIO(content)):
        key = binding.key
        if key not in values:
            updated_parts.append(binding.original.string)
            continue
        updated_parts.append(_leading_line_breaks(binding.original.string))
        if key in remaining:
            updated_parts.append(_format_assignment(key, remaining.pop(key)))

    updated_content = "".join(updated_parts)
    if remaining:
        if updated_content and not updated_content.endswith("\n"):
            updated_content = f"{updated_content}\n"
        updated_content += "".join(
            _format_assignment(key, value) for key, value in sorted(remaining.items())
        )

    mkdir(path.parent, parents=True, exist_ok=True)
    write_text_atomic(path, updated_content)


def set_dotenv_value(path: Path, key: str, value: str) -> None:
    """Set *key* to *value* in a dotenv file."""

    set_dotenv_values(path, {key: value})
