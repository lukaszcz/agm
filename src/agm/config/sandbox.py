"""Sandbox settings resolution and merge helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

from agm.project.layout import project_config_dir, project_deps_dir, project_notes_dir

JsonDict = dict[str, object]


def json_dict(value: object) -> JsonDict:
    if isinstance(value, dict):
        return cast(JsonDict, value)
    return {}


def load_settings(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        raw: object = json.load(handle)
    return json_dict(raw)


def merge_settings_chain(settings_list: list[JsonDict]) -> JsonDict:
    if not settings_list:
        return {}
    merged = settings_list[0]
    for settings in settings_list[1:]:
        merged = merge_settings(merged, settings)
    return merged


def merge_settings(home_data: JsonDict, local_data: JsonDict) -> JsonDict:
    merged = dict(home_data)
    if "enabled" in local_data and local_data["enabled"] is not None:
        merged["enabled"] = local_data["enabled"]
    if isinstance(local_data.get("network"), dict):
        local_network = cast(JsonDict, local_data["network"])
        home_network = (
            cast(JsonDict, home_data["network"])
            if isinstance(home_data.get("network"), dict)
            else {}
        )
        merged["network"] = {**home_network, **local_network}
    if isinstance(local_data.get("filesystem"), dict):
        local_filesystem = cast(JsonDict, local_data["filesystem"])
        home_filesystem = (
            cast(JsonDict, home_data["filesystem"])
            if isinstance(home_data.get("filesystem"), dict)
            else {}
        )
        merged["filesystem"] = {**home_filesystem, **local_filesystem}
    if isinstance(local_data.get("ignoreViolations"), dict):
        merged["ignoreViolations"] = local_data["ignoreViolations"]
    if (
        "enableWeakerNestedSandbox" in local_data
        and local_data["enableWeakerNestedSandbox"] is not None
    ):
        merged["enableWeakerNestedSandbox"] = local_data["enableWeakerNestedSandbox"]
    return merged


def patch_for_proj_dir(settings: JsonDict, proj_dir: Path) -> JsonDict:
    patched = dict(settings)
    filesystem_value = patched.get("filesystem")
    if isinstance(filesystem_value, dict):
        filesystem = cast(JsonDict, filesystem_value)
    else:
        filesystem = {}
        patched["filesystem"] = filesystem
    allow_write_value = filesystem.get("allowWrite")
    if isinstance(allow_write_value, list):
        allow_write = [entry for entry in allow_write_value if isinstance(entry, str)]
    else:
        allow_write = []
    filesystem["allowWrite"] = allow_write
    for path in (project_notes_dir(proj_dir), project_deps_dir(proj_dir)):
        path_str = str(path)
        if path_str not in allow_write:
            allow_write.append(path_str)
    return patched


def sandbox_settings_path(
    settings_dir: Path, command_name: str, alias_command_name: str | None = None
) -> Path:
    for candidate_name in [command_name, alias_command_name]:
        if candidate_name is None:
            continue
        executable_name = Path(candidate_name).name or candidate_name
        command_settings = settings_dir / f"{executable_name}.json"
        if command_settings.is_file():
            return command_settings
    return settings_dir / "default.json"


def sandbox_settings_candidates(
    *,
    cwd: Path,
    home: Path,
    proj_dir: Path | None,
    command_name: str,
    alias_command_name: str | None = None,
) -> list[Path]:
    candidates = [
        sandbox_settings_path(home / ".agm" / "sandbox", command_name, alias_command_name)
    ]
    if proj_dir is not None:
        candidates.append(
            sandbox_settings_path(
                project_config_dir(proj_dir) / "sandbox", command_name, alias_command_name
            )
        )
    candidates.append(sandbox_settings_path(cwd / ".sandbox", command_name, alias_command_name))
    return candidates


def _normalize_path(path_pattern: str, cwd: Path) -> Path:
    if path_pattern == "~":
        normalized = Path.home()
    elif path_pattern.startswith("~/"):
        normalized = Path(path_pattern).expanduser()
    elif os.path.isabs(path_pattern):
        normalized = Path(path_pattern)
    else:
        normalized = cwd / path_pattern
    return normalized.resolve(strict=False)


def _first_missing_component(target: Path, cwd: Path) -> Path | None:
    try:
        relpath = target.relative_to(cwd)
    except ValueError:
        return None
    current = cwd
    for part in relpath.parts:
        current = current / part
        if not current.exists():
            return current
    return None


def track_bwrap_artifacts(settings_path: Path, cwd: Path) -> list[Path]:
    data = load_settings(settings_path)

    mandatory_deny_paths = [
        ".gitconfig",
        ".gitmodules",
        ".bashrc",
        ".bash_profile",
        ".zshrc",
        ".zprofile",
        ".profile",
        ".ripgreprc",
        ".mcp.json",
        ".vscode",
        ".idea",
        ".claude/commands",
        ".claude/agents",
    ]
    if (cwd / ".git").is_dir():
        mandatory_deny_paths.extend([".git/hooks", ".git/config"])

    deny_write: list[str] = []
    filesystem_value = data.get("filesystem")
    if isinstance(filesystem_value, dict):
        filesystem = cast(JsonDict, filesystem_value)
        raw_deny_write = filesystem.get("denyWrite")
        if isinstance(raw_deny_write, list):
            deny_write_candidates = cast(list[object], raw_deny_write)
            deny_write = [
                entry
                for entry in deny_write_candidates
                if isinstance(entry, str) and not any(ch in entry for ch in "*?[]")
            ]

    seen: set[Path] = set()
    tracked: list[Path] = []
    for candidate in [*mandatory_deny_paths, *deny_write]:
        normalized = _normalize_path(candidate, cwd)
        missing = _first_missing_component(normalized, cwd)
        if missing is None or missing in seen:
            continue
        seen.add(missing)
        tracked.append(missing)
    return tracked
