"""General TOML-backed AGM configuration helpers."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Generic, TypeVar

from agm.config.engine_keys import ENGINE_KEY_KINDS, PATH_ENGINE_KEYS
from agm.config.home import agm_home_dir, agm_path_candidates
from agm.core.fs import mkdir, write_text
from agm.core.parse import parse_timeout as parse_timeout
from agm.core.toml import (
    TomlDict,
    dumps_toml,
    empty_toml_doc,
    load_toml_doc,
    load_toml_file,
    set_toml_table_value,
    toml_dict,
)
from agm.project.layout import project_config_dir
from agm.util.interp import interp_preserving

_T = TypeVar("_T")
_K = TypeVar("_K")
_D = TypeVar("_D")


@dataclass(frozen=True)
class GeneralConfig:
    """Path-normalized general configuration and its precedence-ordered layers.

    ``layers`` run from least to most specific and retain the provenance that
    qualified AgL config lookup needs when equivalent table aliases differ
    between files. ``merged`` preserves the ordinary key-wise view consumed by
    existing general-config readers.
    """

    layers: tuple[TomlDict, ...]
    merged: TomlDict

    @classmethod
    def from_layers(cls, layers: Iterable[TomlDict]) -> GeneralConfig:
        """Build a config view from normalized layers in precedence order."""

        ordered_layers = tuple(layers)
        merged: TomlDict = {}
        for layer in ordered_layers:
            merged = _merge_config(merged, layer)
        return cls(layers=ordered_layers, merged=merged)


# Known path-like fields per config section. Values interpolate ``%{name}``
# environment variables, expand ``~``, and resolve against the config file's directory
# before merging, so that relative paths are always interpreted relative to
# the config file that defines them.  When the config-dir-resolved path does
# not exist, cwd is used as a fallback.
_CONFIG_PATH_FIELDS: dict[str, Sequence[str]] = {
    # ``[exec]`` carries exactly the path-valued engine keys.
    "exec": PATH_ENGINE_KEYS,
}


def _merge_config(base: TomlDict, override: TomlDict) -> TomlDict:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_config(toml_dict(existing), toml_dict(value))
            continue
        merged[key] = value
    return merged


def config_file_candidates(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> list[Path]:
    candidates = agm_path_candidates(home=home, relative_path=Path("config.toml"), env=env)
    if proj_dir is not None:
        candidates.append(project_config_dir(proj_dir) / "config.toml")
    candidates.append(cwd / ".agm" / "config.toml")
    return candidates


@dataclass(frozen=True)
class CommandSetting(Generic[_T]):
    """One ``[run]`` setting: a default plus its per-command overrides."""

    default: _T
    overrides: Mapping[str, _T]

    def for_command(self, command_name: str) -> _T:
        return self.overrides.get(command_name, self.default)


@dataclass(frozen=True)
class RunConfig:
    """Resolved run-command configuration: one shape per ``[run]`` setting."""

    alias: CommandSetting[str | None]
    memory: CommandSetting[str | None]
    swap: CommandSetting[str | None]
    pty: CommandSetting[bool]

    def alias_for(self, command_name: str) -> str | None:
        return self.alias.for_command(command_name)

    def memory_limit_for(self, command_name: str | None) -> str | None:
        if command_name is None:
            return self.memory.default
        return self.memory.for_command(command_name)

    def swap_limit_for(self, command_name: str | None) -> str | None:
        return self.swap.default if command_name is None else self.swap.for_command(command_name)

    def pty_for(self, command_name: str) -> bool:
        return self.pty.for_command(command_name)


def _interpolate_and_expand_section_paths(
    section: TomlDict,
    fields: Sequence[str],
) -> tuple[TomlDict, set[str]]:
    resolved = dict(section)
    unresolved_fields: set[str] = set()
    for field in fields:
        value = resolved.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        interpolated, unresolved = interp_preserving(value, os.environ)
        resolved[field] = os.path.expanduser(interpolated)
        if unresolved:
            unresolved_fields.add(field)
    return resolved, unresolved_fields


def _anchor_section_paths(
    section: TomlDict,
    fields: Sequence[str],
    config_dir: Path,
    cwd: Path,
    *,
    unresolved_fields: set[str],
) -> TomlDict:
    resolved = dict(section)
    for field in fields:
        value = resolved.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        # Values with unresolved holes are not usable paths, so leave them
        # unanchored. Interpolation has already run exactly once above.
        if field in unresolved_fields:
            continue
        path = Path(value)
        if path.is_absolute():
            continue
        config_resolved = (config_dir / path).resolve()
        if config_resolved.exists():
            resolved[field] = str(config_resolved)
        else:
            resolved[field] = str((cwd / path).resolve())
    return resolved


def resolve_section_paths(
    section: TomlDict,
    fields: Sequence[str],
    config_dir: Path,
    cwd: Path,
) -> TomlDict:
    """Interpolate and anchor *section*'s path-like *fields* to *config_dir*.

    Shared by config-file loading (anchored to the file's own directory) and
    package manifest ``[config]`` engine-key resolution (anchored to *cwd* for
    both arguments, like a CLI flag value), so each origin resolves a relative
    path the way its own spelling implies.
    """
    expanded, unresolved_fields = _interpolate_and_expand_section_paths(section, fields)
    resolved = _anchor_section_paths(
        expanded,
        fields,
        config_dir,
        cwd,
        unresolved_fields=unresolved_fields,
    )
    for key, value in resolved.items():
        if isinstance(value, dict) and key not in fields:
            resolved[key] = resolve_section_paths(toml_dict(value), fields, config_dir, cwd)
    return resolved


def _resolve_config_file_paths(config: TomlDict, config_dir: Path, cwd: Path) -> TomlDict:
    resolved = dict(config)
    for section_name, section in resolved.items():
        if isinstance(section, dict):
            # Every known section lists its path-like fields explicitly;
            # unknown/program sections carry only engine keys, so they fall
            # back to the path-valued ones.
            fields = _CONFIG_PATH_FIELDS.get(section_name, PATH_ENGINE_KEYS)
            resolved[section_name] = resolve_section_paths(
                toml_dict(section),
                fields,
                config_dir,
                cwd,
            )
    return resolved


def load_general_config(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> GeneralConfig:
    """Load path-normalized config layers and their conventional merged view."""

    layers: list[TomlDict] = []
    for path in config_file_candidates(home=home, proj_dir=proj_dir, cwd=cwd, env=env):
        if path.is_file():
            raw = load_toml_file(path)
            layers.append(_resolve_config_file_paths(raw, config_dir=path.parent, cwd=cwd))
    return GeneralConfig.from_layers(layers)


def load_merged_config(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> TomlDict:
    """Load the conventional merged config view.

    Kept for readers that do not need per-file provenance. Qualified config
    consumers must use :func:`load_general_config` and retain its layers.
    """

    return load_general_config(home=home, proj_dir=proj_dir, cwd=cwd, env=env).merged


def _setting_value(table: TomlDict, key: str, kind: type[_K]) -> _K | None:
    """Return ``table[key]`` when it is a *kind* value, treating empty text as absent."""
    value = table.get(key)
    if isinstance(value, str) and not value:
        return None
    return value if isinstance(value, kind) else None


def _command_setting(
    run_table: TomlDict,
    key: str,
    kind: type[_K],
    *,
    default: _D,
    global_key: bool = True,
) -> CommandSetting[_K | _D]:
    """Load one ``[run]`` setting from its bare key and its per-command tables.

    The default is ``[run].<key>`` when *global_key*, falling back to
    *default*; ``[run.<command>].<key>`` overrides it for that command.
    """
    resolved: _K | _D = default
    if global_key:
        global_value = _setting_value(run_table, key, kind)
        if global_value is not None:
            resolved = global_value
    overrides: dict[str, _K | _D] = {}
    for command_name, command_config in run_table.items():
        value = _setting_value(toml_dict(command_config), key, kind)
        if value is not None:
            overrides[command_name] = value
    return CommandSetting(default=resolved, overrides=overrides)


def load_run_config(*, home: Path, proj_dir: Path | None, cwd: Path) -> RunConfig:
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    run_table = toml_dict(merged.get("run"))
    return RunConfig(
        # A bare ``[run] alias`` would rename every command, so alias is
        # per-command only.
        alias=_command_setting(run_table, "alias", str, default=None, global_key=False),
        memory=_command_setting(run_table, "memory", str, default=None),
        swap=_command_setting(run_table, "swap", str, default=None),
        pty=_command_setting(run_table, "pty", bool, default=True),
    )


def _optional_str(table: TomlDict, key: str) -> str | None:
    value = table.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _optional_positive_int(table: TomlDict, key: str) -> int | None:
    """Return a positive int config value, or ``None`` when absent/invalid.

    ``None`` lets the caller apply its own canonical default rather than baking a
    default into the config layer.
    """
    value = table.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _optional_bool(table: TomlDict, key: str, *, default: bool = False) -> bool:
    value = table.get(key)
    return value if isinstance(value, bool) else default


def _optional_timeout(table: TomlDict, key: str) -> float | None:
    value = table.get(key)
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool) and value > 0:
        return float(value)
    if isinstance(value, str) and value.strip():
        return parse_timeout(value)
    return None


@dataclass(frozen=True)
class ExecConfig:
    """Resolved exec-command configuration."""

    # Raw TOML values (``None`` = unset): exec/repl decode the bool and
    # ``parse_error_retries`` engine keys through the shared host-value decoder.
    strict_json: object | None
    timeout: float | None
    trace: object | None
    trace_file: str | None
    # Echo streamed agent output on the host's stderr channel.
    echo: bool = False
    # Raw TOML value (a string or a native table): exec/repl decode it as a
    # host Agent value through the shared host-value decoder.
    default_agent: object | None = None
    # Raw TOML value (a string or a native table): exec/repl decode it as a
    # host AgentSandbox value through the shared host-value decoder.
    default_sandbox: object | None = None
    # Optional recursion call-depth override (None = use the canonical default).
    max_call_depth: int | None = None
    # Keep debugging artifacts (``std/fs`` temporary paths) past the host session.
    debug: object | None = None
    parse_error_retries: object | None = None


def exec_config_from_merged(
    merged: TomlDict,
    *,
    program_table: dict[str, object] | None = None,
    package_table: dict[str, object] | None = None,
) -> ExecConfig:
    """Build :class:`ExecConfig` from an already-merged config dict.

    Every exec host already holds a merged config, so the ``[exec]`` section is
    derived from it rather than re-read. ``[exec]`` has no named sub-tables: a
    per-program override comes from the program's own qualified table, because
    an AgL program is identified by its module route rather than by a flat
    name the way ``[run.<command>]``'s settings are.

    When *program_table* is supplied, each engine key present in that already
    resolved qualified program table overrides the global ``[exec]`` value.
    *package_table* — a package-owned program's manifest ``[config]`` values —
    ranks between ``[exec]`` and *program_table*. Engine keys use kebab-case
    names: ``strict-json``, ``trace-file``.
    """
    exec_table = toml_dict(merged.get("exec"))

    # Effective precedence, low to high: [exec] < package manifest < the
    # qualified program table (exact route, then inherited groups).
    effective: TomlDict = dict(exec_table)
    for table in (package_table, program_table):
        if table is None:
            continue
        for key, _ in ENGINE_KEY_KINDS:
            if key in table:
                effective[key] = table[key]

    resolved_strict_json = effective.get("strict-json")
    resolved_echo = _optional_bool(exec_table, "echo") is True
    resolved_max_call_depth = _optional_positive_int(exec_table, "max-call-depth")

    resolved_timeout = _optional_timeout(effective, "timeout")

    resolved_trace = effective.get("trace")
    resolved_trace_file = _optional_str(effective, "trace-file")
    resolved_debug = effective.get("debug")
    # Keep every explicitly supplied engine value raw (``default-agent`` and
    # ``default-sandbox`` may be a native TOML table) so exec/repl decode and
    # validate it through the shared host-value decoder at their AgL host
    # boundary; other commands stay free of AgL imports.
    resolved_default_agent = effective.get("default-agent")
    resolved_default_sandbox = effective.get("default-sandbox")
    resolved_parse_error_retries = effective.get("parse-error-retries")

    return ExecConfig(
        echo=resolved_echo,
        strict_json=resolved_strict_json,
        max_call_depth=resolved_max_call_depth,
        timeout=resolved_timeout,
        trace=resolved_trace,
        trace_file=resolved_trace_file,
        default_agent=resolved_default_agent,
        default_sandbox=resolved_default_sandbox,
        debug=resolved_debug,
        parse_error_retries=resolved_parse_error_retries,
    )


@dataclass(frozen=True)
class ReplConfig:
    """Resolved REPL configuration."""

    theme: str
    echo: bool = True
    echo_unit: bool = False


def load_repl_config(*, home: Path, proj_dir: Path | None, cwd: Path) -> ReplConfig:
    """Load ``[repl]`` configuration, merging all config layers."""
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    section = toml_dict(merged.get("repl", {}))
    theme = section.get("theme", "auto")
    if not isinstance(theme, str) or theme not in ("dark", "light", "auto"):
        theme = "auto"
    return ReplConfig(
        theme=theme,
        echo=_optional_bool(section, "echo", default=True),
        echo_unit=_optional_bool(section, "echo-unit", default=False),
    )


def save_repl_setting(key: str, value: str | bool, *, home: Path) -> None:
    """Persist one ``[repl]`` setting (``theme``, ``echo``, ``echo-unit``, …) to config.toml.

    Single generic save path for every REPL setting: writes the raw TOML
    value as-is (a ``bool`` round-trips as a TOML boolean, a ``str`` as a TOML
    string), always to the home-level ``config.toml``.
    """
    path = agm_home_dir(home=home) / "config.toml"
    doc = load_toml_doc(path) if path.is_file() else empty_toml_doc()
    set_toml_table_value(doc, "repl", key, value)
    mkdir(path.parent, parents=True, exist_ok=True)
    write_text(path, dumps_toml(doc), encoding="utf-8")
