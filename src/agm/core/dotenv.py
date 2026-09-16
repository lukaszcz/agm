"""Dotenv file editing helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from agm.core.fs import mkdir, write_text_atomic

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


def write_dotenv_values(path: Path, values: Mapping[str, str]) -> None:
    """Replace *path* atomically with sorted dotenv assignments."""

    content = "".join(_format_assignment(key, value) for key, value in sorted(values.items()))
    mkdir(path.parent, parents=True, exist_ok=True)
    write_text_atomic(path, content)
