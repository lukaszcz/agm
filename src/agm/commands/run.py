"""agm run."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from agm.commands.args import RunArgs
from agm.config.general import load_run_config
from agm.config.sandbox import (
    JsonDict,
    load_settings,
    merge_settings_chain,
    patch_for_proj_dir,
    sandbox_settings_candidates,
    track_bwrap_artifacts,
)

DEFAULT_MEMORY_LIMIT = "20G"


def normalize_run_command(run_command: list[str]) -> list[str]:
    if run_command[:1] == ["--"]:
        return run_command[1:]
    return run_command


def _parse_memory_limit_value(limit: str) -> int | None:
    stripped = limit.strip()
    if not stripped:
        return None
    sign = 1
    body = stripped
    if body[:1] in {"+", "-"}:
        if body[0] == "-":
            sign = -1
        body = body[1:]
    digits = ""
    index = 0
    while index < len(body) and body[index].isdigit():
        digits += body[index]
        index += 1
    if not digits:
        return None
    suffix = body[index:].upper()
    multipliers = {
        "": 1,
        "B": 1,
        "K": 1000,
        "KB": 1000,
        "M": 1000**2,
        "MB": 1000**2,
        "G": 1000**3,
        "GB": 1000**3,
        "T": 1000**4,
        "TB": 1000**4,
        "P": 1000**5,
        "PB": 1000**5,
        "E": 1000**6,
        "EB": 1000**6,
    }
    multiplier = multipliers.get(suffix)
    if multiplier is None:
        return None
    return sign * int(digits) * multiplier


def _memory_limit_enabled(limit: str | None) -> bool:
    if limit is None:
        return False
    parsed = _parse_memory_limit_value(limit)
    if parsed is None:
        return True
    return parsed > 0


def _systemd_run_prefix(limit: str) -> list[str]:
    return ["systemd-run", "--user", "--scope", "-p", f"MemoryMax={limit}"]



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
    run_command = normalize_run_command(list(run_args.run_command))
    if not run_command:
        print("Error: command is required.", file=sys.stderr)
        raise SystemExit(1)

    run_config = load_run_config(
        home=Path(resolved_env["HOME"]),
        proj_dir=Path(resolved_env["PROJ_DIR"]) if resolved_env.get("PROJ_DIR") else None,
        cwd=current,
    )
    command_name = Path(run_command[0]).name or run_command[0]
    command_alias = run_config.alias_for(command_name)
    configured_memory_limit = run_config.memory_limit_for(command_name)
    if run_args.no_sandbox:
        effective_memory_limit = run_args.memory
    else:
        effective_memory_limit = run_args.memory or configured_memory_limit or DEFAULT_MEMORY_LIMIT
    effective_run_command = list(run_command)
    if command_alias is not None:
        effective_run_command[0] = command_alias
    if not run_args.no_sandbox and shutil.which("srt", path=resolved_env.get("PATH")) is None:
        print("Error: srt is not installed or not in PATH.", file=sys.stderr)
        print("Install it with: npm install -g @anthropic-ai/sandbox-runtime", file=sys.stderr)
        raise SystemExit(1)
    if _memory_limit_enabled(effective_memory_limit) and (
        shutil.which("systemd-run", path=resolved_env.get("PATH")) is None
    ):
        print("Error: systemd-run is not installed or not in PATH.", file=sys.stderr)
        raise SystemExit(1)

    temp_files: list[Path] = []
    tracked_artifacts: list[Path] = []
    try:
        if run_args.no_sandbox:
            subprocess_args = list(effective_run_command)
        else:
            if run_args.settings_file is not None:
                selected_settings = Path(run_args.settings_file)
                if not selected_settings.is_file():
                    print(
                        f"Error: settings file not found: {run_args.settings_file}",
                        file=sys.stderr,
                    )
                    raise SystemExit(1)
            else:
                settings_candidates = sandbox_settings_candidates(
                    cwd=current,
                    home=Path(resolved_env["HOME"]),
                    proj_dir=(
                        Path(resolved_env["PROJ_DIR"]) if resolved_env.get("PROJ_DIR") else None
                    ),
                    command_name=run_command[0],
                    alias_command_name=(
                        effective_run_command[0] if command_alias is not None else None
                    ),
                )
                found_settings = [path for path in settings_candidates if path.is_file()]
                if not found_settings:
                    print("Error: no sandbox settings file found.", file=sys.stderr)
                    print(
                        "Checked: " + ", ".join(str(path) for path in settings_candidates),
                        file=sys.stderr,
                    )
                    raise SystemExit(1)
                if len(found_settings) == 1:
                    selected_settings = found_settings[0]
                else:
                    settings_data = [
                        load_settings(settings_path) for settings_path in found_settings
                    ]
                    selected_settings = _write_json_temp(
                        merge_settings_chain(settings_data), temp_files
                    )

            if not run_args.no_patch and resolved_env.get("PROJ_DIR"):
                selected_settings_data = load_settings(selected_settings)
                selected_settings = _write_json_temp(
                    patch_for_proj_dir(selected_settings_data, Path(resolved_env["PROJ_DIR"])),
                    temp_files,
                )

            tracked_artifacts = track_bwrap_artifacts(selected_settings, current)
            subprocess_args = [
                "srt",
                "--settings",
                str(selected_settings),
                "--",
                *effective_run_command,
            ]
        if _memory_limit_enabled(effective_memory_limit):
            assert effective_memory_limit is not None
            subprocess_args = [*_systemd_run_prefix(effective_memory_limit), *subprocess_args]
        raise SystemExit(
            subprocess.run(
                subprocess_args,
                cwd=current,
                env=resolved_env,
                check=False,
            ).returncode,
        )
    finally:
        _cleanup(temp_files, tracked_artifacts)
