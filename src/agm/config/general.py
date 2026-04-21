"""General TOML-backed AGM configuration helpers."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agm.core.env import agm_installation_prefix
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
    install_prefix = agm_installation_prefix()
    global_config = home / ".agm" / "config.toml"
    if install_prefix is not None:
        install_config = install_prefix / ".agm" / "config.toml"
        if install_config.is_file():
            global_config = install_config

    candidates = [global_config]
    if proj_dir is not None:
        candidates.append(project_config_dir(proj_dir) / "config.toml")
    candidates.append(cwd / ".agm" / "config.toml")
    return candidates


@dataclass(frozen=True)
class RunConfig:
    """Resolved run-command configuration."""

    aliases: dict[str, str]
    default_memory_limit: str | None
    command_memory_limits: dict[str, str]

    def alias_for(self, command_name: str) -> str | None:
        return self.aliases.get(command_name)

    def memory_limit_for(self, command_name: str) -> str | None:
        return self.command_memory_limits.get(command_name, self.default_memory_limit)


@dataclass(frozen=True)
class LoopConfig:
    """Resolved loop-command configuration."""

    command: str | None
    tasks_dir: str | None


def load_merged_config(*, home: Path, proj_dir: Path | None, cwd: Path) -> TomlDict:
    merged: TomlDict = {}
    for path in config_file_candidates(home=home, proj_dir=proj_dir, cwd=cwd):
        if path.is_file():
            merged = _merge_config(merged, _load_config_file(path))
    return merged


def load_run_config(*, home: Path, proj_dir: Path | None, cwd: Path) -> RunConfig:
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    run_table = _toml_dict(merged.get("run"))
    aliases: dict[str, str] = {}
    command_memory_limits: dict[str, str] = {}
    default_memory = run_table.get("memory")
    default_memory_limit = (
        default_memory if isinstance(default_memory, str) and default_memory else None
    )
    for command_name, command_config in run_table.items():
        config = _toml_dict(command_config)
        alias = config.get("alias")
        if isinstance(alias, str) and alias:
            aliases[command_name] = alias
        memory = config.get("memory")
        if isinstance(memory, str) and memory:
            command_memory_limits[command_name] = memory
    return RunConfig(
        aliases=aliases,
        default_memory_limit=default_memory_limit,
        command_memory_limits=command_memory_limits,
    )


def load_loop_config(*, home: Path, proj_dir: Path | None, cwd: Path) -> LoopConfig:
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    loop_table = _toml_dict(merged.get("loop"))
    command = loop_table.get("command")
    tasks_dir = loop_table.get("tasks_dir")
    resolved_command = command if isinstance(command, str) and command.strip() else None
    resolved_tasks_dir = tasks_dir if isinstance(tasks_dir, str) and tasks_dir.strip() else None
    return LoopConfig(command=resolved_command, tasks_dir=resolved_tasks_dir)
