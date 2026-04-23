"""SRT sandbox execution orchestration."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from agm.config.sandbox.srt import (
    JsonDict,
    load_settings,
    merge_settings_chain,
    patch_for_proj_dir,
    sandbox_settings_candidates,
    track_bwrap_artifacts,
)
from agm.core import dry_run
from agm.core.fs import is_dir, is_file, rmdir, stat, unlink
from agm.core.process import run_foreground


def require_srt_installed(path: str | None) -> None:
    if shutil.which("srt", path=path) is not None:
        return
    print("Error: srt is not installed or not in PATH.", file=sys.stderr)
    print("Install it with: npm install -g @anthropic-ai/sandbox-runtime", file=sys.stderr)
    raise SystemExit(1)


def _write_json_temp(data: JsonDict, temp_files: list[Path]) -> Path:
    with NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        json.dump(data, handle)
        path = Path(handle.name)
    temp_files.append(path)
    return path


def _cleanup(temp_files: list[Path], tracked_artifacts: list[Path]) -> None:
    for temp_file in temp_files:
        try:
            unlink(temp_file)
        except FileNotFoundError:
            pass
    for artifact in tracked_artifacts:
        try:
            if is_file(artifact) and stat(artifact).st_size == 0:
                unlink(artifact)
            elif is_dir(artifact):
                rmdir(artifact)
        except OSError:
            pass


def _resolve_settings_path(
    *,
    cwd: Path,
    home: Path,
    proj_dir: Path | None,
    command_name: str,
    alias_command_name: str | None,
    settings_file: str | None,
    temp_files: list[Path],
) -> Path:
    if settings_file is not None:
        selected_settings = Path(settings_file)
        if not is_file(selected_settings):
            print(f"Error: settings file not found: {settings_file}", file=sys.stderr)
            raise SystemExit(1)
        return selected_settings

    settings_candidates = sandbox_settings_candidates(
        cwd=cwd,
        home=home,
        proj_dir=proj_dir,
        command_name=command_name,
        alias_command_name=alias_command_name,
    )
    found_settings = [path for path in settings_candidates if is_file(path)]
    if not found_settings:
        print("Error: no sandbox settings file found.", file=sys.stderr)
        print("Checked: " + ", ".join(str(path) for path in settings_candidates), file=sys.stderr)
        raise SystemExit(1)
    if len(found_settings) == 1:
        return found_settings[0]
    settings_data = [load_settings(settings_path) for settings_path in found_settings]
    return _write_json_temp(merge_settings_chain(settings_data), temp_files)


def _print_dry_run(
    *,
    cwd: Path,
    home: Path,
    proj_dir: Path | None,
    command: list[str],
    command_name: str,
    alias_command_name: str | None,
    settings_file: str | None,
    patch_proj_dir: Path | None,
    process_prefix: list[str],
) -> None:
    if settings_file is not None:
        settings_source = "explicit"
        settings_detail = settings_file
    else:
        settings_candidates = sandbox_settings_candidates(
            cwd=cwd,
            home=home,
            proj_dir=proj_dir,
            command_name=command_name,
            alias_command_name=alias_command_name,
        )
        settings_source = "merged"
        settings_detail = ", ".join(str(path) for path in settings_candidates)

    dry_run.print_configuration("sandbox")
    dry_run.print_detail("settings source", settings_source)
    dry_run.print_detail("settings candidates", settings_detail)
    dry_run.print_detail(
        "patch proj dir path",
        str(patch_proj_dir) if patch_proj_dir is not None else "disabled",
    )

    dry_run.print_labeled_command(
        "sandbox",
        [*process_prefix, "srt", "--settings", "<dry-run-settings>", "--", *command],
        cwd=cwd,
    )


def run_sandboxed(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    home: Path,
    proj_dir: Path | None,
    command_name: str,
    alias_command_name: str | None,
    settings_file: str | None,
    patch_proj_dir: Path | None,
    process_prefix: list[str] | None = None,
    interrupt_cleanup_cmd: list[str] | None = None,
) -> None:
    require_srt_installed(env.get("PATH"))
    resolved_process_prefix = [] if process_prefix is None else list(process_prefix)

    if dry_run.enabled():
        _print_dry_run(
            cwd=cwd,
            home=home,
            proj_dir=proj_dir,
            command=command,
            command_name=command_name,
            alias_command_name=alias_command_name,
            settings_file=settings_file,
            patch_proj_dir=patch_proj_dir,
            process_prefix=resolved_process_prefix,
        )
        return

    temp_files: list[Path] = []
    tracked_artifacts: list[Path] = []
    try:
        selected_settings = _resolve_settings_path(
            cwd=cwd,
            home=home,
            proj_dir=proj_dir,
            command_name=command_name,
            alias_command_name=alias_command_name,
            settings_file=settings_file,
            temp_files=temp_files,
        )
        if patch_proj_dir is not None:
            selected_settings_data = load_settings(selected_settings)
            selected_settings = _write_json_temp(
                patch_for_proj_dir(selected_settings_data, patch_proj_dir),
                temp_files,
            )

        tracked_artifacts = track_bwrap_artifacts(selected_settings, cwd)
        subprocess_args = [
            *resolved_process_prefix,
            "srt",
            "--settings",
            str(selected_settings),
            "--",
            *command,
        ]
        try:
            raise SystemExit(
                run_foreground(
                    subprocess_args,
                    cwd=cwd,
                    env=env,
                    interrupt_cleanup_cmd=interrupt_cleanup_cmd,
                )
            )
        except KeyboardInterrupt:
            print("\nInterrupted")
            raise SystemExit(130)
    finally:
        _cleanup(temp_files, tracked_artifacts)
