"""agm run: a thin client of the sandbox preparation library."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import NoReturn

from agm.cli_support.args import RunArgs
from agm.config.context import current_config_context
from agm.config.general import load_run_config
from agm.core import dry_run
from agm.core.env import clone_env
from agm.core.path import display_path
from agm.core.process import run_foreground
from agm.sandbox.backend import SandboxSettingsError, SandboxUnavailableError, default_backend
from agm.sandbox.prepare import (
    ResolvedLimits,
    prepare,
    print_dry_run,
    resolve_limits,
    settings_source,
)
from agm.sandbox.profile import profile_name
from agm.sandbox.request import Default, LimitSpec, SandboxRequest, SandboxSpec


def normalize_run_command(run_command: list[str]) -> list[str]:
    if run_command[:1] == ["--"]:
        return run_command[1:]
    return run_command


def _limit_spec(*, flag_value: str | None, no_limit: bool, no_sandbox: bool) -> LimitSpec:
    """Map `--memory`/`--swap`-style flags to a `LimitSpec`.

    An explicit, non-empty flag always wins (an empty value, e.g. `--memory
    ""`, is treated as not given); otherwise `--no-sandbox` means no default
    limit (today's behaviour: only `--no-sandbox` plus an explicit flag
    applies one), while sandboxed runs fall through to config/built-in
    defaults via `Default`.
    """

    if no_limit:
        return None
    if flag_value:
        return flag_value
    if no_sandbox:
        return None
    return Default


def _limit_detail(raw: str | None) -> str:
    return raw if raw is not None else "disabled"


def _print_run_configuration(
    *,
    request: SandboxRequest,
    command_name: str,
    command_alias: str | None,
    allocate_pty: bool,
    effective_pty: bool,
    limits: ResolvedLimits,
) -> None:
    dry_run.print_configuration("run")
    dry_run.print_detail("cwd", str(request.cwd))
    dry_run.print_detail("sandbox", "enabled" if request.sandboxed else "disabled")
    dry_run.print_detail("patch proj dir", "enabled" if request.spec.patch else "disabled")
    dry_run.print_detail("command name", command_name)
    dry_run.print_detail("alias command", command_alias or "disabled")
    dry_run.print_detail(
        "pty",
        "enabled" if allocate_pty else "disabled" if not effective_pty else "not a terminal",
    )
    dry_run.print_detail("memory limit", _limit_detail(limits.memory_raw))
    dry_run.print_detail("swap limit", _limit_detail(limits.swap_raw))


def _print_sandbox_settings_detail(request: SandboxRequest) -> None:
    spec = request.spec
    if spec.settings_file is not None:
        detail = display_path(spec.settings_file, cwd=request.cwd)
    else:
        candidates = default_backend().settings_candidates(request)
        detail = ", ".join(display_path(path, cwd=request.cwd) for path in candidates)
    patch_target = request.proj_dir if spec.patch else None

    dry_run.print_configuration("sandbox")
    dry_run.print_detail("settings source", settings_source(spec))
    dry_run.print_detail("settings candidates", detail)
    dry_run.print_detail(
        "patch proj dir path",
        display_path(patch_target, cwd=request.cwd) if patch_target is not None else "disabled",
    )


def _report_settings_error(error: SandboxSettingsError, *, cwd: Path) -> None:
    if error.path is not None:
        print(
            f"Error: settings file not found: {display_path(error.path, cwd=cwd)}", file=sys.stderr
        )
    elif error.candidates:
        print("Error: no sandbox settings file found.", file=sys.stderr)
        print(
            "Checked: " + ", ".join(display_path(path, cwd=cwd) for path in error.candidates),
            file=sys.stderr,
        )
    else:
        print(f"Error: {error}", file=sys.stderr)


def _fail_sandbox_error(
    error: SandboxUnavailableError | SandboxSettingsError, *, cwd: Path
) -> NoReturn:
    if isinstance(error, SandboxSettingsError):
        _report_settings_error(error, cwd=cwd)
    else:
        print(f"Error: {error}", file=sys.stderr)
        if error.detail is not None:
            print(error.detail, file=sys.stderr)
    raise SystemExit(1) from error


def run(args: RunArgs) -> None:
    current = Path.cwd()
    resolved_env = clone_env()
    context = current_config_context(cwd=current, env=resolved_env)
    run_command = normalize_run_command(list(args.run_command))
    if not run_command:
        print("Error: command is required.", file=sys.stderr)
        raise SystemExit(1)

    run_config = load_run_config(home=context.home, proj_dir=context.proj_dir, cwd=context.cwd)
    command_name = profile_name(run_command[0])
    command_alias = run_config.alias_for(command_name)

    effective_run_command = list(run_command)
    alias_name: str | None = None
    if command_alias is not None:
        alias_parts = shlex.split(command_alias)
        effective_run_command = [*alias_parts, *effective_run_command[1:]]
        alias_name = profile_name(alias_parts[0])

    configured_pty = run_config.pty_for(command_name)
    effective_pty = configured_pty if args.pty is None else args.pty
    allocate_pty = effective_pty and os.isatty(0) and os.isatty(1)

    spec = SandboxSpec(
        profile_name=command_name,
        memory=_limit_spec(
            flag_value=args.memory, no_limit=args.no_memory_limit, no_sandbox=args.no_sandbox
        ),
        swap=_limit_spec(
            flag_value=args.swap, no_limit=args.no_swap_limit, no_sandbox=args.no_sandbox
        ),
        settings_file=Path(args.settings_file) if args.settings_file is not None else None,
        patch=not args.no_patch,
    )
    request = SandboxRequest(
        command=effective_run_command,
        cwd=current,
        env=resolved_env,
        home=context.home,
        proj_dir=context.proj_dir,
        spec=spec,
        alias_name=alias_name,
        pty=allocate_pty,
        sandboxed=not args.no_sandbox,
    )

    if dry_run.enabled():
        try:
            limits = resolve_limits(spec, run_config)
        except SandboxSettingsError as error:
            _fail_sandbox_error(error, cwd=current)
        _print_run_configuration(
            request=request,
            command_name=command_name,
            command_alias=command_alias,
            allocate_pty=allocate_pty,
            effective_pty=effective_pty,
            limits=limits,
        )
        if request.sandboxed:
            _print_sandbox_settings_detail(request)
        try:
            prepared = prepare(request, run_config=run_config, resolve_settings=False)
        except (SandboxUnavailableError, SandboxSettingsError) as error:
            _fail_sandbox_error(error, cwd=current)
        print_dry_run(prepared, label="sandbox" if request.sandboxed else "run")
        prepared.close()
        return

    try:
        prepared = prepare(request, run_config=run_config)
    except (SandboxUnavailableError, SandboxSettingsError) as error:
        _fail_sandbox_error(error, cwd=current)

    try:
        exit_code = run_foreground(
            prepared.argv,
            cwd=prepared.cwd,
            env=prepared.env,
            interrupt_cleanup_cmd=prepared.interrupt_cleanup_cmd,
            isolate_process_group=True,
        )
    except KeyboardInterrupt:
        print("\nInterrupted")
        exit_code = 130
    finally:
        prepared.close()
    raise SystemExit(exit_code)
