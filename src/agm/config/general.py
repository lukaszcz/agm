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


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def agm_path_candidates(*, home: Path, relative_path: Path) -> list[Path]:
    candidates: list[Path] = []
    install_prefix = agm_installation_prefix()
    if install_prefix is not None:
        candidates.append(install_prefix / ".agm" / relative_path)
    candidates.append(home / ".agm" / relative_path)
    return _unique_paths(candidates)


def resolve_agm_path(*, home: Path, relative_path: Path) -> Path:
    candidates = agm_path_candidates(home=home, relative_path=relative_path)
    for candidate in reversed(candidates):
        if candidate.is_file():
            return candidate
    return candidates[-1]


def config_file_candidates(*, home: Path, proj_dir: Path | None, cwd: Path) -> list[Path]:
    candidates = agm_path_candidates(home=home, relative_path=Path("config.toml"))
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

    runner: str | None
    selector: str | None
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


def load_loop_config(
    *, home: Path, proj_dir: Path | None, cwd: Path, command_name: str | None = None
) -> LoopConfig:
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    loop_table = _toml_dict(merged.get("loop"))
    selected_loop_table = loop_table
    if command_name is not None:
        command_table = _toml_dict(loop_table.get(command_name))
        selected_loop_table = _merge_config(loop_table, command_table)
    runner = selected_loop_table.get("runner")
    selector = selected_loop_table.get("selector")
    tasks_dir = selected_loop_table.get("tasks_dir")
    resolved_runner = runner if isinstance(runner, str) and runner.strip() else None
    resolved_selector = selector if isinstance(selector, str) and selector.strip() else None
    resolved_tasks_dir = tasks_dir if isinstance(tasks_dir, str) and tasks_dir.strip() else None
    return LoopConfig(
        runner=resolved_runner,
        selector=resolved_selector,
        tasks_dir=resolved_tasks_dir,
    )
