"""Shared TOML parsing helpers."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

TomlDict = dict[str, object]


def toml_dict(value: object) -> TomlDict:
    """Coerce *value* to a ``TomlDict``, returning an empty dict for non-dicts."""

    if isinstance(value, dict):
        return cast(TomlDict, value)
    return {}


def load_toml_file(path: Path) -> TomlDict:
    """Parse a TOML file and return its contents as a ``TomlDict``."""

    with path.open("rb") as handle:
        raw: object = tomllib.load(handle)
    return toml_dict(raw)
