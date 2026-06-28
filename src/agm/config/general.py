"""General TOML-backed AGM configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from agm.core.env import agm_installation_prefix
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


class ConfigCommandNotFound(ValueError):
    """Raised when a named command config table is required but missing."""

    def __init__(self, *, section_name: str, command_name: str) -> None:
        self.section_name = section_name
        self.command_name = command_name
        super().__init__(
            f"{section_name} subcommand {command_name!r} is not defined in config"
        )


# Known path-like fields per config section.  Values for these fields are
# expanded (env vars, ~) and resolved against the config file's directory
# before merging, so that relative paths are always interpreted relative to
# the config file that defines them.  When the config-dir-resolved path does
# not exist, cwd is used as a fallback.
_CONFIG_PATH_FIELDS: dict[str, list[str]] = {
    "exec": [
        "log_file",
    ],
    "loop": [
        "tasks_dir",
        "prompt_file",
        "selector_prompt_file",
        "extra_prompt_file",
        "extra_selector_prompt_file",
    ],
    "review": [
        "prompt_file",
        "extra_prompt_file",
        "review_file",
    ],
    "revise": [
        "prompt_file",
        "extra_prompt_file",
    ],
    "refine": [
        "review_prompt_file",
        "extra_review_prompt_file",
        "revise_prompt_file",
        "extra_revise_prompt_file",
    ],
}

_CONFIG_PATH_SENTINELS: dict[str, dict[str, set[str]]] = {
    "review": {
        "review_file": {"auto", "none"},
    },
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


def resolve_default_prompt_file(filename: str, *, home: Path) -> Path:
    """Resolve a default prompt file from the AGM prompt directory."""

    return resolve_agm_path(home=home, relative_path=Path("prompts") / filename)


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
    default_swap_limit: str | None
    command_swap_limits: dict[str, str]

    def alias_for(self, command_name: str) -> str | None:
        return self.aliases.get(command_name)

    def memory_limit_for(self, command_name: str) -> str | None:
        return self.command_memory_limits.get(command_name, self.default_memory_limit)

    def swap_limit_for(self, command_name: str) -> str | None:
        return self.command_swap_limits.get(command_name, self.default_swap_limit)


@dataclass(frozen=True)
class LoopConfig:
    """Resolved loop-command configuration."""

    runner: str | None
    selector: str | None
    no_selector: bool
    tasks_dir: str | None
    prompt: str | None
    prompt_file: str | None
    selector_prompt: str | None
    selector_prompt_file: str | None
    extra_prompt: str | None
    extra_prompt_file: str | None
    extra_selector_prompt: str | None
    extra_selector_prompt_file: str | None
    timeout: float | None


@dataclass(frozen=True)
class ReviewConfig:
    """Resolved review-command configuration."""

    runner: str | None
    scope: str | None
    aspects: str | None
    extra_aspects: str | None
    prompt: str | None
    prompt_file: str | None
    extra_prompt: str | None
    extra_prompt_file: str | None
    review_file: str | None


@dataclass(frozen=True)
class ReviseConfig:
    """Resolved revise-command configuration."""

    runner: str | None
    prompt: str | None
    prompt_file: str | None
    extra_prompt: str | None
    extra_prompt_file: str | None


@dataclass(frozen=True)
class RefineConfig:
    """Resolved refine-command configuration."""

    max_steps: int | None
    no_max_steps: bool
    runner: str | None
    reviewer: str | None
    reviser: str | None
    scope: str | None
    aspects: str | None
    review_prompt: str | None
    review_prompt_file: str | None
    extra_review_prompt: str | None
    extra_review_prompt_file: str | None
    revise_prompt: str | None
    revise_prompt_file: str | None
    extra_revise_prompt: str | None
    extra_revise_prompt_file: str | None
    save_review: bool


def _resolve_section_paths(
    section: TomlDict,
    fields: list[str],
    config_dir: Path,
    cwd: Path,
    *,
    sentinels: dict[str, set[str]],
) -> TomlDict:
    resolved = dict(section)
    for field in fields:
        value = resolved.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        expanded = os.path.expanduser(os.path.expandvars(value))
        if expanded in sentinels.get(field, set()):
            resolved[field] = expanded
            continue
        path = Path(expanded)
        if path.is_absolute():
            resolved[field] = expanded
            continue
        config_resolved = (config_dir / path).resolve()
        if config_resolved.exists():
            resolved[field] = str(config_resolved)
        else:
            resolved[field] = str((cwd / path).resolve())
    for key, value in resolved.items():
        if isinstance(value, dict) and key not in fields:
            resolved[key] = _resolve_section_paths(
                toml_dict(value),
                fields,
                config_dir,
                cwd,
                sentinels=sentinels,
            )
    return resolved


def _resolve_config_file_paths(config: TomlDict, config_dir: Path, cwd: Path) -> TomlDict:
    resolved = dict(config)
    for section_name, fields in _CONFIG_PATH_FIELDS.items():
        section = resolved.get(section_name)
        if isinstance(section, dict):
            resolved[section_name] = _resolve_section_paths(
                toml_dict(section),
                fields,
                config_dir,
                cwd,
                sentinels=_CONFIG_PATH_SENTINELS.get(section_name, {}),
            )
    return resolved


def load_merged_config(*, home: Path, proj_dir: Path | None, cwd: Path) -> TomlDict:
    merged: TomlDict = {}
    for path in config_file_candidates(home=home, proj_dir=proj_dir, cwd=cwd):
        if path.is_file():
            raw = load_toml_file(path)
            resolved = _resolve_config_file_paths(raw, config_dir=path.parent, cwd=cwd)
            merged = _merge_config(merged, resolved)
    return merged


def load_run_config(*, home: Path, proj_dir: Path | None, cwd: Path) -> RunConfig:
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    run_table = toml_dict(merged.get("run"))
    aliases: dict[str, str] = {}
    command_memory_limits: dict[str, str] = {}
    command_swap_limits: dict[str, str] = {}
    default_memory = run_table.get("memory")
    default_memory_limit = (
        default_memory if isinstance(default_memory, str) and default_memory else None
    )
    default_swap = run_table.get("swap")
    default_swap_limit = default_swap if isinstance(default_swap, str) and default_swap else None
    for command_name, command_config in run_table.items():
        config = toml_dict(command_config)
        alias = config.get("alias")
        if isinstance(alias, str) and alias:
            aliases[command_name] = alias
        memory = config.get("memory")
        if isinstance(memory, str) and memory:
            command_memory_limits[command_name] = memory
        swap = config.get("swap")
        if isinstance(swap, str) and swap:
            command_swap_limits[command_name] = swap
    return RunConfig(
        aliases=aliases,
        default_memory_limit=default_memory_limit,
        command_memory_limits=command_memory_limits,
        default_swap_limit=default_swap_limit,
        command_swap_limits=command_swap_limits,
    )


def load_loop_config(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    command_name: str | None = None,
    require_command: bool = False,
) -> LoopConfig:
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    return loop_config_from_merged(
        merged, command_name=command_name, require_command=require_command
    )


def loop_config_from_merged(
    merged: TomlDict,
    *,
    command_name: str | None = None,
    require_command: bool = False,
) -> LoopConfig:
    """Build :class:`LoopConfig` from an already-merged config dict.

    Split out from :func:`load_loop_config` so a caller that already holds a
    merged config (e.g. ``agm exec`` resolving its default runner) can derive
    the ``[loop]`` section without re-reading and re-merging the files.
    """
    selected_loop_table = _select_command_table(
        toml_dict(merged.get("loop")),
        section_name="loop",
        command_name=command_name,
        require_command=require_command,
    )
    runner = selected_loop_table.get("runner")
    selector = selected_loop_table.get("selector")
    no_selector_raw = selected_loop_table.get("no_selector")
    tasks_dir = selected_loop_table.get("tasks_dir")
    resolved_runner = runner if isinstance(runner, str) and runner.strip() else None
    resolved_selector = selector if isinstance(selector, str) and selector.strip() else None
    resolved_no_selector = bool(no_selector_raw) if isinstance(no_selector_raw, bool) else False
    resolved_tasks_dir = tasks_dir if isinstance(tasks_dir, str) and tasks_dir.strip() else None
    prompt = selected_loop_table.get("prompt")
    prompt_file = selected_loop_table.get("prompt_file")
    resolved_prompt = prompt if isinstance(prompt, str) and prompt.strip() else None
    resolved_prompt_file = (
        prompt_file if isinstance(prompt_file, str) and prompt_file.strip() else None
    )
    selector_prompt = selected_loop_table.get("selector_prompt")
    selector_prompt_file = selected_loop_table.get("selector_prompt_file")
    resolved_selector_prompt = (
        selector_prompt if isinstance(selector_prompt, str) and selector_prompt.strip() else None
    )
    resolved_selector_prompt_file = (
        selector_prompt_file
        if isinstance(selector_prompt_file, str) and selector_prompt_file.strip()
        else None
    )
    extra_prompt = selected_loop_table.get("extra_prompt")
    extra_prompt_file = selected_loop_table.get("extra_prompt_file")
    resolved_extra_prompt = (
        extra_prompt if isinstance(extra_prompt, str) and extra_prompt.strip() else None
    )
    resolved_extra_prompt_file = (
        extra_prompt_file
        if isinstance(extra_prompt_file, str) and extra_prompt_file.strip()
        else None
    )
    extra_selector_prompt = selected_loop_table.get("extra_selector_prompt")
    extra_selector_prompt_file = selected_loop_table.get("extra_selector_prompt_file")
    resolved_extra_selector_prompt = (
        extra_selector_prompt
        if isinstance(extra_selector_prompt, str) and extra_selector_prompt.strip()
        else None
    )
    resolved_extra_selector_prompt_file = (
        extra_selector_prompt_file
        if isinstance(extra_selector_prompt_file, str) and extra_selector_prompt_file.strip()
        else None
    )
    resolved_timeout = _optional_timeout(selected_loop_table, "timeout")
    return LoopConfig(
        runner=resolved_runner,
        selector=resolved_selector,
        no_selector=resolved_no_selector,
        tasks_dir=resolved_tasks_dir,
        prompt=resolved_prompt,
        prompt_file=resolved_prompt_file,
        selector_prompt=resolved_selector_prompt,
        selector_prompt_file=resolved_selector_prompt_file,
        extra_prompt=resolved_extra_prompt,
        extra_prompt_file=resolved_extra_prompt_file,
        extra_selector_prompt=resolved_extra_selector_prompt,
        extra_selector_prompt_file=resolved_extra_selector_prompt_file,
        timeout=resolved_timeout,
    )


def _optional_str(table: TomlDict, key: str) -> str | None:
    value = table.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _optional_positive_int_or_unlimited(table: TomlDict, key: str) -> int | None:
    value = table.get(key)
    if isinstance(value, str) and value.strip().lower() == "unlimited":
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _optional_bool(table: TomlDict, key: str, *, default: bool = False) -> bool:
    value = table.get(key)
    return value if isinstance(value, bool) else default


def _optional_positive_int(table: TomlDict, key: str, *, default: int) -> int:
    value = table.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


def _optional_timeout(table: TomlDict, key: str) -> float | None:
    value = table.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    if isinstance(value, str) and value.strip():
        return parse_timeout(value)
    return None


def _select_command_table(
    table: TomlDict,
    *,
    section_name: str,
    command_name: str | None,
    require_command: bool,
) -> TomlDict:
    if command_name is None:
        return table
    command_table = table.get(command_name)
    if isinstance(command_table, dict):
        return _merge_config(table, toml_dict(command_table))
    if require_command:
        raise ConfigCommandNotFound(section_name=section_name, command_name=command_name)
    return table


def load_review_config(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    command_name: str | None = None,
    require_command: bool = True,
) -> ReviewConfig:
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    table = _select_command_table(
        toml_dict(merged.get("review")),
        section_name="review",
        command_name=command_name,
        require_command=require_command,
    )
    return ReviewConfig(
        runner=_optional_str(table, "runner"),
        scope=_optional_str(table, "scope"),
        aspects=_optional_str(table, "aspects"),
        extra_aspects=_optional_str(table, "extra_aspects"),
        prompt=_optional_str(table, "prompt"),
        prompt_file=_optional_str(table, "prompt_file"),
        extra_prompt=_optional_str(table, "extra_prompt"),
        extra_prompt_file=_optional_str(table, "extra_prompt_file"),
        review_file=_optional_str(table, "review_file"),
    )


def load_revise_config(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    command_name: str | None = None,
    require_command: bool = True,
) -> ReviseConfig:
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    table = _select_command_table(
        toml_dict(merged.get("revise")),
        section_name="revise",
        command_name=command_name,
        require_command=require_command,
    )
    return ReviseConfig(
        runner=_optional_str(table, "runner"),
        prompt=_optional_str(table, "prompt"),
        prompt_file=_optional_str(table, "prompt_file"),
        extra_prompt=_optional_str(table, "extra_prompt"),
        extra_prompt_file=_optional_str(table, "extra_prompt_file"),
    )


@dataclass(frozen=True)
class ExecConfig:
    """Resolved exec-command configuration."""

    runner: str | None
    strict_json: bool
    default_loop_limit: int
    timeout: float | None
    agents: dict[str, str]
    log: bool
    log_file: str | None


def load_exec_config(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    command_name: str | None = None,
) -> ExecConfig:
    """Load and resolve the ``[exec]`` configuration section.

    Follows the same layering pattern as ``load_loop_config``:
    - home/.agm/config.toml
    - project config/config.toml
    - cwd/.agm/config.toml

    When ``command_name`` is provided, the ``[exec.<command_name>]`` sub-table
    is merged over the base ``[exec]`` table.  The name ``agents`` is reserved
    for the structural ``[exec.agents]`` map and is never treated as a
    per-command override sub-table.
    """
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    return exec_config_from_merged(merged, command_name=command_name)


def exec_config_from_merged(
    merged: TomlDict, *, command_name: str | None = None
) -> ExecConfig:
    """Build :class:`ExecConfig` from an already-merged config dict.

    Split out from :func:`load_exec_config` so a caller that already holds a
    merged config (e.g. ``agm exec``, which also needs ``[params.<key>]``) can
    derive the ``[exec]`` section without re-reading and re-merging the files.
    """
    # ``agents`` is a reserved structural sub-table, not a workflow override:
    # selecting it as a command would merge the agent map in as scalar config.
    selected_command = None if command_name == "agents" else command_name
    exec_table = _select_command_table(
        toml_dict(merged.get("exec")),
        section_name="exec",
        command_name=selected_command,
        require_command=False,
    )

    resolved_runner = _optional_str(exec_table, "runner")
    resolved_strict_json = _optional_bool(exec_table, "strict_json")
    resolved_loop_limit = _optional_positive_int(exec_table, "default_loop_limit", default=5)

    resolved_timeout = _optional_timeout(exec_table, "timeout")

    agents_raw = exec_table.get("agents")
    resolved_agents: dict[str, str] = {}
    if isinstance(agents_raw, dict):
        for k, v in agents_raw.items():
            if isinstance(k, str) and isinstance(v, str) and v.strip():
                resolved_agents[k] = v

    resolved_log = _optional_bool(exec_table, "log")
    resolved_log_file = _optional_str(exec_table, "log_file")

    return ExecConfig(
        runner=resolved_runner,
        strict_json=resolved_strict_json,
        default_loop_limit=resolved_loop_limit,
        timeout=resolved_timeout,
        agents=resolved_agents,
        log=resolved_log,
        log_file=resolved_log_file,
    )


def load_refine_config(
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
    command_name: str | None = None,
    require_command: bool = True,
) -> RefineConfig:
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    table = _select_command_table(
        toml_dict(merged.get("refine")),
        section_name="refine",
        command_name=command_name,
        require_command=require_command,
    )
    return RefineConfig(
        max_steps=_optional_positive_int_or_unlimited(table, "max_steps"),
        no_max_steps=_optional_bool(table, "no_max_steps"),
        runner=_optional_str(table, "runner"),
        reviewer=_optional_str(table, "reviewer"),
        reviser=_optional_str(table, "reviser"),
        scope=_optional_str(table, "scope"),
        aspects=_optional_str(table, "aspects"),
        review_prompt=_optional_str(table, "review_prompt"),
        review_prompt_file=_optional_str(table, "review_prompt_file"),
        extra_review_prompt=_optional_str(table, "extra_review_prompt"),
        extra_review_prompt_file=_optional_str(table, "extra_review_prompt_file"),
        revise_prompt=_optional_str(table, "revise_prompt"),
        revise_prompt_file=_optional_str(table, "revise_prompt_file"),
        extra_revise_prompt=_optional_str(table, "extra_revise_prompt"),
        extra_revise_prompt_file=_optional_str(table, "extra_revise_prompt_file"),
        save_review=_optional_bool(table, "save_review", default=True),
    )


def load_params_config(
    program_name: str,
    *,
    home: Path,
    proj_dir: Path | None,
    cwd: Path,
) -> dict[str, object]:
    """Return the ``[params.<program_name>]`` table from the merged config.

    Loads the merged config across all config-file layers (home → project → cwd)
    and returns the ``[params.<program_name>]`` sub-table as a dict of TOML-native
    values.  Returns an empty dict when the table is absent.

    Values are returned as-is (TOML-native Python objects) so that
    ``convert_param_value`` in the runtime can coerce them to AgL types.  In
    particular, ``bool`` values must NOT be stringified (``str(True)`` → ``"True"``
    which the bool coercion path rejects).

    Note for decimal params: TOML has no decimal type; a TOML float (e.g.
    ``3.14``) is a Python ``float`` and AgL rejects binary floats.  Decimal param
    values in config must be written as TOML strings (e.g. ``ratio = "3.14"``);
    ``convert_param_value`` will parse them correctly via ``json.loads(parse_float=Decimal)``.
    """
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    return params_config_from_merged(merged, program_name)


def params_config_from_merged(
    merged: TomlDict, program_name: str
) -> dict[str, object]:
    """Return the ``[params.<program_name>]`` table from an already-merged config.

    Split out from :func:`load_params_config` so ``agm exec`` can derive the
    param table from the same merged config it already loaded for ``[exec]``,
    rather than re-reading and re-merging every config file.
    """
    params_section = toml_dict(merged.get("params"))
    program_table = params_section.get(program_name)
    if isinstance(program_table, dict):
        return dict(program_table)
    return {}


@dataclass(frozen=True)
class ReplConfig:
    """Resolved REPL configuration."""

    theme: str


def load_repl_config(*, home: Path, proj_dir: Path | None, cwd: Path) -> ReplConfig:
    """Load ``[repl]`` configuration, merging all config layers."""
    merged = load_merged_config(home=home, proj_dir=proj_dir, cwd=cwd)
    section = toml_dict(merged.get("repl", {}))
    theme = section.get("theme", "auto")
    if not isinstance(theme, str) or theme not in ("dark", "light", "auto"):
        theme = "auto"
    return ReplConfig(theme=theme)


def save_repl_theme(theme: str, *, home: Path) -> None:
    """Persist the REPL theme preference to the home-level ``config.toml``."""
    path = home / ".agm" / "config.toml"
    doc = load_toml_doc(path) if path.is_file() else empty_toml_doc()
    set_toml_table_value(doc, "repl", "theme", theme)
    mkdir(path.parent, parents=True, exist_ok=True)
    write_text(path, dumps_toml(doc), encoding="utf-8")
