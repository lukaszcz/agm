"""agm run."""

from __future__ import annotations

import os
import shlex
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
DEFAULT_SWAP_LIMIT = "0"
_SYSTEMD_DELEGATED_CGROUP_BOOTSTRAP = (
    'CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup); '
    'mkdir -p "${CG}/init"; '
    'echo $$ > "${CG}/init/cgroup.procs"; '
    'echo "+memory" > "${CG}/cgroup.subtree_control"; '
    'export SANDBOX_CGROUP="$CG"; '
    'exec "$@"'
)


def normalize_run_command(run_command: list[str]) -> list[str]:
    if run_command[:1] == ["--"]:
        return run_command[1:]
    return run_command


def _normalize_systemd_limit(limit: str) -> str:
    if limit.strip().lower() == "unlimited":
        return "infinity"
    return limit


def _systemd_run_prefix(*, memory_limit: str | None, swap_limit: str | None) -> list[str]:
    prefix = [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
    ]
    if memory_limit is not None:
        prefix.extend(["-p", f"MemoryMax={_normalize_systemd_limit(memory_limit)}"])
    if swap_limit is not None:
        prefix.extend(["-p", f"MemorySwapMax={_normalize_systemd_limit(swap_limit)}"])
    prefix.extend(["-p", "Delegate=yes"])
    return prefix


def _systemd_scope_name() -> str:
    return f"agm-run-{uuid4().hex}.scope"


def _resource_limit_run_context(
    env: dict[str, str], memory_limit: str | None, swap_limit: str | None
) -> tuple[list[str], list[str] | None]:
    if memory_limit is None and swap_limit is None:
        return [], None
    if shutil.which("systemd-run", path=env.get("PATH")) is None:
        print("Error: systemd-run is not installed or not in PATH.", file=sys.stderr)
        raise SystemExit(1)
    scope_name = _systemd_scope_name()
    return (
        [
            *_systemd_run_prefix(memory_limit=memory_limit, swap_limit=swap_limit),
            "--unit",
            scope_name,
            "--",
            "bash",
            "-c",
            _SYSTEMD_DELEGATED_CGROUP_BOOTSTRAP,
            "--",
        ],
        ["systemctl", "--user", "stop", scope_name],
    )


def _run_with_optional_resource_limits(
    *,
    subprocess_args: list[str],
    cwd: Path,
    env: dict[str, str],
    memory_limit: str | None,
    swap_limit: str | None,
) -> None:
    process_prefix, interrupt_cleanup_cmd = _resource_limit_run_context(
        env, memory_limit, swap_limit
    )
    subprocess_args = [*process_prefix, *subprocess_args]
    try:
        raise SystemExit(
            run_foreground(
                subprocess_args,
                cwd=cwd,
                env=env,
                interrupt_cleanup_cmd=interrupt_cleanup_cmd,
                isolate_process_group=True,
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
    configured_swap_limit = run_config.swap_limit_for(command_name)
    if run_args.no_memory_limit:
        effective_memory_limit = None
    elif run_args.no_sandbox:
        effective_memory_limit = run_args.memory
    else:
        effective_memory_limit = run_args.memory or configured_memory_limit or DEFAULT_MEMORY_LIMIT
    if run_args.no_swap_limit:
        effective_swap_limit = None
    elif run_args.no_sandbox:
        effective_swap_limit = run_args.swap
    else:
        effective_swap_limit = run_args.swap or configured_swap_limit or DEFAULT_SWAP_LIMIT
    effective_run_command = list(run_command)
    if command_alias is not None:
        alias_parts = shlex.split(command_alias)
        effective_run_command = [*alias_parts, *effective_run_command[1:]]
    process_prefix, interrupt_cleanup_cmd = _resource_limit_run_context(
        resolved_env, effective_memory_limit, effective_swap_limit
    )
    if dry_run.enabled():
        dry_run.print_configuration("run")
        dry_run.print_detail("cwd", str(current))
        dry_run.print_detail("sandbox", "disabled" if run_args.no_sandbox else "enabled")
        dry_run.print_detail("patch proj dir", "disabled" if run_args.no_patch else "enabled")
        dry_run.print_detail("command name", command_name)
        dry_run.print_detail("alias command", command_alias or "disabled")
        dry_run.print_detail(
            "memory limit",
            effective_memory_limit if effective_memory_limit is not None else "disabled",
        )
        dry_run.print_detail(
            "swap limit", effective_swap_limit if effective_swap_limit is not None else "disabled"
        )
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
        dry_run.print_labeled_command("run", subprocess_args, cwd=current)
        return

    if run_args.no_sandbox:
        _run_with_optional_resource_limits(
            subprocess_args=list(effective_run_command),
            cwd=current,
            env=resolved_env,
            memory_limit=effective_memory_limit,
            swap_limit=effective_swap_limit,
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
