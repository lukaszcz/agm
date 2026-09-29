"""Configured default reasoning effort for agent specifications.

Host policy applied when an agent is used, never when its AgL value is built:
a spec with empty ``thinking`` takes the ``effort`` of its most specific
``[agent.<cli>...]`` table that sets one. ``AgentCommand`` and a non-empty
``thinking`` are never touched; an explicit ``effort = ""`` or no configured
effort leaves the agent CLI's own default.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TypeAlias

from agm.agent.spec import AgentClaude, AgentCodex, AgentCommand, AgentPi, AgentSpec
from agm.core.toml import TomlDict, toml_dict

__all__ = [
    "AgentConfig",
    "AgentConfigError",
    "AgentSpecResolver",
    "default_effort_resolver",
    "with_default_effort",
]

#: Maps the spec an agent value decodes to onto the spec actually dispatched.
AgentSpecResolver: TypeAlias = Callable[[AgentSpec], AgentSpec]

#: How many name levels each CLI's table nests below ``[agent.<cli>]``.
_CLI_NAME_LEVELS: Mapping[str, int] = MappingProxyType({"claude": 1, "codex": 1, "pi": 2})


class AgentConfigError(ValueError):
    """The ``[agent]`` config section is malformed."""


@dataclass(frozen=True)
class AgentConfig:
    """The validated ``[agent]`` section."""

    table: TomlDict

    @classmethod
    def from_merged(cls, merged: TomlDict) -> AgentConfig:
        """Validate the ``[agent]`` section of *merged* config, or raise `AgentConfigError`."""
        table = toml_dict(merged.get("agent"))
        for cli, value in table.items():
            if cli not in _CLI_NAME_LEVELS or not isinstance(value, dict):
                raise AgentConfigError(f"agent.{cli} must be one of the tables {_cli_names()}")
            _validate_table(toml_dict(value), ("agent", cli), _CLI_NAME_LEVELS[cli])
        return cls(table)

    def effort(self, path: tuple[str, ...]) -> str | None:
        """``effort`` of the ``[agent.<path>]`` table, or ``None`` when it sets none."""
        table = self.table
        for key in path:
            table = toml_dict(table.get(key))
        effort = table.get("effort")
        return effort if isinstance(effort, str) else None


def _cli_names() -> str:
    return ", ".join(sorted(_CLI_NAME_LEVELS))


def _validate_table(table: TomlDict, path: tuple[str, ...], levels: int) -> None:
    """Allow only text ``effort`` and at most *levels* further nested name tables."""
    for key, value in table.items():
        key_path = ".".join((*path, key))
        if isinstance(value, dict):
            if levels == 0:
                raise AgentConfigError(f"{key_path} nests too deeply")
            _validate_table(toml_dict(value), (*path, key), levels - 1)
        elif key != "effort":
            raise AgentConfigError(f"{key_path} is not a known setting")
        elif not isinstance(value, str):
            raise AgentConfigError(f"{key_path} must be text")


def with_default_effort(spec: AgentSpec, config: AgentConfig) -> AgentSpec:
    """*spec* with its configured default effort when its ``thinking`` is empty."""
    if isinstance(spec, AgentCommand) or spec.thinking:
        return spec
    for path in _effort_table_paths(spec):
        effort = config.effort(path)
        if effort is not None:
            return replace(spec, thinking=effort)
    return spec


def _effort_table_paths(spec: AgentClaude | AgentCodex | AgentPi) -> list[tuple[str, ...]]:
    """``[agent]`` table paths for *spec*, most specific first; empty names address none."""
    paths: list[tuple[str, ...]]
    if isinstance(spec, AgentPi):
        paths = [("pi", spec.provider, spec.model), ("pi", spec.provider), ("pi",)]
    else:
        cli = "claude" if isinstance(spec, AgentClaude) else "codex"
        paths = [(cli, spec.model), (cli,)]
    return [path for path in paths if all(path)]


def default_effort_resolver(merged: TomlDict) -> AgentSpecResolver:
    """The `with_default_effort` resolver for *merged* config; raises `AgentConfigError`."""
    config = AgentConfig.from_merged(merged)
    return lambda spec: with_default_effort(spec, config)
