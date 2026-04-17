"""General TOML-backed AGM configuration helpers."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agm.project.layout import project_config_dir

TomlDict = dict[str, object]


def _toml_dict(value: object) -> TomlDict:
    if isinstance(value, dict):
        return cast(TomlDict, value)
    return {}


def _merge_config(base: TomlDict, override: TomlDict) -> TomlDict:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_config(_toml_dict(existing), _toml_dict(value))
            continue
        merged[key] = value
    return merged


def _load_config_file(path: Path) -> TomlDict:
    with path.open("rb") as handle:
        raw: object = tomllib.load(handle)
    return _toml_dict(raw)


def config_file_candidates(*, home: Path, proj_dir: Path | None, cwd: Path) -> list[Path]:
    candidates = [home / ".agm" / "config.toml"]
    if proj_dir is not None:
        candidates.append(project_config_dir(proj_dir) / "config.toml")
    candidates.append(cwd / ".agm" / "config.toml")
    return candidates


@dataclass(frozen=True)
class RunConfig:
    """Resolved run-command configuration."""

    aliases: dict[str, str]

    def alias_for(self, command_name: str) -> str | None:
        return self.aliases.get(command_name)


def load_run_config(*, home: Path, proj_dir: Path | None, cwd: Path) -> RunConfig:
    merged: TomlDict = {}
    for path in config_file_candidates(home=home, proj_dir=proj_dir, cwd=cwd):
        if path.is_file():
            merged = _merge_config(merged, _load_config_file(path))

    run_table = _toml_dict(merged.get("run"))
    aliases: dict[str, str] = {}
    for command_name, command_config in run_table.items():
        config = _toml_dict(command_config)
        alias = config.get("alias")
        if isinstance(alias, str) and alias:
            aliases[command_name] = alias
    return RunConfig(aliases=aliases)
