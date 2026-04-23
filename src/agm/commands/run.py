"""agm run."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from agm.commands.args import RunArgs
from agm.config.general import load_run_config
from agm.core import dry_run
from agm.core.process import run_foreground
from agm.sandbox import srt

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


def _systemd_scope_name() -> str:
    return f"agm-run-{uuid4().hex}.scope"


def _memory_limit_run_context(
    env: dict[str, str], memory_limit: str | None
) -> tuple[list[str], list[str] | None]:
    if not _memory_limit_enabled(memory_limit):
        return [], None
    if shutil.which("systemd-run", path=env.get("PATH")) is None:
        print("Error: systemd-run is not installed or not in PATH.", file=sys.stderr)
        raise SystemExit(1)
    assert memory_limit is not None
    scope_name = _systemd_scope_name()
    return (
        [*_systemd_run_prefix(memory_limit), "--unit", scope_name],
        ["systemctl", "--user", "stop", scope_name],
    )


def _run_with_optional_memory_limit(
    *,
    subprocess_args: list[str],
    cwd: Path,
    env: dict[str, str],
    memory_limit: str | None,
) -> None:
    process_prefix, interrupt_cleanup_cmd = _memory_limit_run_context(env, memory_limit)
    subprocess_args = [*process_prefix, *subprocess_args]
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
    process_prefix, interrupt_cleanup_cmd = _memory_limit_run_context(
        resolved_env, effective_memory_limit
    )
    if dry_run.enabled():
        if not run_args.no_sandbox:
            srt.run_sandboxed(
                command=effective_run_command,
                cwd=current,
                env=resolved_env,
                home=Path(resolved_env["HOME"]),
                proj_dir=Path(resolved_env["PROJ_DIR"]) if resolved_env.get("PROJ_DIR") else None,
                command_name=run_command[0],
                alias_command_name=effective_run_command[0] if command_alias is not None else None,
                settings_file=run_args.settings_file,
                patch_proj_dir=(
                    Path(resolved_env["PROJ_DIR"])
                    if not run_args.no_patch and resolved_env.get("PROJ_DIR")
                    else None
                ),
                process_prefix=process_prefix,
            )
            return
        subprocess_args = [*process_prefix, *effective_run_command]
        dry_run.print_command(subprocess_args, cwd=current)
        return

    if run_args.no_sandbox:
        _run_with_optional_memory_limit(
            subprocess_args=list(effective_run_command),
            cwd=current,
            env=resolved_env,
            memory_limit=effective_memory_limit,
        )
        return

    srt.run_sandboxed(
        command=effective_run_command,
        cwd=current,
        env=resolved_env,
        home=Path(resolved_env["HOME"]),
        proj_dir=Path(resolved_env["PROJ_DIR"]) if resolved_env.get("PROJ_DIR") else None,
        command_name=run_command[0],
        alias_command_name=effective_run_command[0] if command_alias is not None else None,
        settings_file=run_args.settings_file,
        patch_proj_dir=(
            Path(resolved_env["PROJ_DIR"])
            if not run_args.no_patch and resolved_env.get("PROJ_DIR")
            else None
        ),
        process_prefix=process_prefix,
        interrupt_cleanup_cmd=interrupt_cleanup_cmd,
    )
