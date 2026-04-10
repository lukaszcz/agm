"""Sandbox runtime wrapper."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

JsonDict = dict[str, object]


def usage() -> str:
    return """Usage:
  sandbox.sh [--no-patch] [-f settings.json] <command> [args...]

Runs a command inside Anthropic Sandbox Runtime (`srt`).

If `-f` is not provided, settings are loaded from:
  1. ~/.sandbox/default.json
  2. ./.sandbox/default.json

If both files exist, the project-local file overrides the home file with
section-aware merging.

If `PROJ_DIR` is set, the selected settings file is patched temporarily to add
`$PROJ_DIR` to `filesystem.allowWrite`.
Use `--no-patch` to disable that behavior.
"""


def merge_settings(home_data: JsonDict, local_data: JsonDict) -> JsonDict:
    """Merge settings with the same field semantics as sandbox.sh."""

    merged = dict(home_data)
    if "enabled" in local_data and local_data["enabled"] is not None:
        merged["enabled"] = local_data["enabled"]
    if isinstance(local_data.get("network"), dict):
        local_network = cast(JsonDict, local_data["network"])
        home_network = cast(JsonDict, home_data["network"]) if isinstance(home_data.get("network"), dict) else {}
        merged["network"] = {
            **home_network,
            **local_network,
        }
    if isinstance(local_data.get("filesystem"), dict):
        local_filesystem = cast(JsonDict, local_data["filesystem"])
        home_filesystem = cast(JsonDict, home_data["filesystem"]) if isinstance(home_data.get("filesystem"), dict) else {}
        merged["filesystem"] = {
            **home_filesystem,
            **local_filesystem,
        }
    if isinstance(local_data.get("ignoreViolations"), dict):
        merged["ignoreViolations"] = local_data["ignoreViolations"]
    if (
        "enableWeakerNestedSandbox" in local_data
        and local_data["enableWeakerNestedSandbox"] is not None
    ):
        merged["enableWeakerNestedSandbox"] = local_data["enableWeakerNestedSandbox"]
    return merged


def patch_for_proj_dir(settings: JsonDict, proj_dir: str) -> JsonDict:
    """Add *proj_dir* to filesystem.allowWrite."""

    patched = dict(settings)
    filesystem = patched.get("filesystem")
    if not isinstance(filesystem, dict):
        filesystem = {}
        patched["filesystem"] = filesystem
    allow_write = filesystem.get("allowWrite")
    if not isinstance(allow_write, list):
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
    normalized = normalized.resolve(strict=False)
    return normalized


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
    """Return filesystem artifacts that may be created by the sandbox."""

    with settings_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)
    data = cast(JsonDict, raw_data) if isinstance(raw_data, dict) else {}

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
    filesystem = data.get("filesystem")
    if isinstance(filesystem, dict):
        raw_deny_write = filesystem.get("denyWrite")
        if isinstance(raw_deny_write, list):
            deny_write = [
                entry
                for entry in raw_deny_write
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


def run_in_sandbox(
    *,
    no_patch: bool,
    settings_file: str | None,
    run_command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command inside srt with resolved sandbox settings."""

    current = Path.cwd() if cwd is None else cwd.resolve()
    resolved_env = dict(os.environ if env is None else env)
    if run_command[:1] == ["--"]:
        run_command = run_command[1:]
    if not run_command:
        print("Error: command is required.", file=sys.stderr)
        print(usage(), file=sys.stderr, end="")
        raise SystemExit(1)
    if shutil.which("srt", path=resolved_env.get("PATH")) is None:
        print("Error: srt is not installed or not in PATH.", file=sys.stderr)
        print("Install it with: npm install -g @anthropic-ai/sandbox-runtime", file=sys.stderr)
        raise SystemExit(1)

    temp_files: list[Path] = []
    tracked_artifacts: list[Path] = []
    try:
        if settings_file is not None:
            selected_settings = Path(settings_file)
            if not selected_settings.is_file():
                print(f"Error: settings file not found: {settings_file}", file=sys.stderr)
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
                try:
                    with home_settings.open("r", encoding="utf-8") as handle:
                        raw_home_data = json.load(handle)
                    with local_settings.open("r", encoding="utf-8") as handle:
                        raw_local_data = json.load(handle)
                    home_data = cast(JsonDict, raw_home_data) if isinstance(raw_home_data, dict) else {}
                    local_data = cast(JsonDict, raw_local_data) if isinstance(raw_local_data, dict) else {}
                except Exception as exc:
                    print(str(exc), file=sys.stderr)
                    raise SystemExit(1) from exc
                selected_settings = _write_json_temp(merge_settings(home_data, local_data), temp_files)

        if not no_patch and resolved_env.get("PROJ_DIR"):
            try:
                with selected_settings.open("r", encoding="utf-8") as handle:
                    raw_settings_data = json.load(handle)
                settings_data = cast(JsonDict, raw_settings_data) if isinstance(raw_settings_data, dict) else {}
            except Exception as exc:
                print(str(exc), file=sys.stderr)
                raise SystemExit(1) from exc
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
