"""agm run."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from agm.commands.args import RunArgs

JsonDict = dict[str, object]


def _json_dict(value: object) -> JsonDict:
    if isinstance(value, dict):
        return cast(JsonDict, value)
    return {}


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


def patch_for_proj_dir(settings: JsonDict, proj_dir: str) -> JsonDict:
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
    if proj_dir not in allow_write:
        allow_write.append(proj_dir)
    return patched


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
    with settings_path.open("r", encoding="utf-8") as handle:
        raw_data: object = json.load(handle)
    data = _json_dict(raw_data)

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


def _write_json_temp(data: JsonDict, temp_files: list[Path]) -> Path:
    with NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        json.dump(data, handle)
        path = Path(handle.name)
    temp_files.append(path)
    return path


def _cleanup(temp_files: list[Path], tracked_artifacts: list[Path]) -> None:
    for temp_file in temp_files:
        try:
            temp_file.unlink()
        except FileNotFoundError:
            pass
    for artifact in tracked_artifacts:
        try:
            if artifact.is_file() and artifact.stat().st_size == 0:
                artifact.unlink()
            elif artifact.is_dir():
                artifact.rmdir()
        except OSError:
            pass


def run(args: RunArgs) -> None:
    current = Path.cwd()
    resolved_env = dict(os.environ)
    run_args = args
    run_command = list(run_args.run_command)
    if run_command[:1] == ["--"]:
        run_command = run_command[1:]
    if not run_command:
        print("Error: command is required.", file=sys.stderr)
        raise SystemExit(1)
    if shutil.which("srt", path=resolved_env.get("PATH")) is None:
        print("Error: srt is not installed or not in PATH.", file=sys.stderr)
        print("Install it with: npm install -g @anthropic-ai/sandbox-runtime", file=sys.stderr)
        raise SystemExit(1)

    temp_files: list[Path] = []
    tracked_artifacts: list[Path] = []
    try:
        if run_args.settings_file is not None:
            selected_settings = Path(run_args.settings_file)
            if not selected_settings.is_file():
                print(
                    f"Error: settings file not found: {run_args.settings_file}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
        else:
            home_settings = Path(resolved_env["HOME"]) / ".sandbox" / "default.json"
            local_settings = current / ".sandbox" / "default.json"
            found_settings = [path for path in (home_settings, local_settings) if path.is_file()]
            if not found_settings:
                print("Error: no sandbox settings file found.", file=sys.stderr)
                print(f"Checked: {home_settings} and {local_settings}", file=sys.stderr)
                raise SystemExit(1)
            if len(found_settings) == 1:
                selected_settings = found_settings[0]
            else:
                with home_settings.open("r", encoding="utf-8") as handle:
                    raw_home_data: object = json.load(handle)
                with local_settings.open("r", encoding="utf-8") as handle:
                    raw_local_data: object = json.load(handle)
                home_data = _json_dict(raw_home_data)
                local_data = _json_dict(raw_local_data)
                selected_settings = _write_json_temp(
                    merge_settings(home_data, local_data), temp_files
                )

        if not run_args.no_patch and resolved_env.get("PROJ_DIR"):
            with selected_settings.open("r", encoding="utf-8") as handle:
                raw_settings_data: object = json.load(handle)
            settings_data = _json_dict(raw_settings_data)
            selected_settings = _write_json_temp(
                patch_for_proj_dir(settings_data, resolved_env["PROJ_DIR"]),
                temp_files,
            )

        tracked_artifacts = track_bwrap_artifacts(selected_settings, current)
        raise SystemExit(
            subprocess.run(
                ["srt", "--settings", str(selected_settings), "--", *run_command],
                cwd=current,
                env=resolved_env,
                check=False,
            ).returncode,
        )
    finally:
        _cleanup(temp_files, tracked_artifacts)
