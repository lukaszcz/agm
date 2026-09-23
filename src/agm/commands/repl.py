"""Implementation of the ``agm repl`` command.

Launches an interactive read-eval-print loop for the AgL workflow language.
The REPL shares ``agm exec``'s ``[exec]`` configuration (default-agent,
default-sandbox, and timeout), so an interactive session evaluates entries
with the same agent dispatch backing a batch ``agm exec`` run would use.

The command itself is thin: it resolves configuration the same way ``exec``
does, builds a value-driven dispatcher, constructs a :class:`ReplSession`,
then hands control to one of two front ends sharing the same UI-free loop
core (:mod:`agm.agl.repl.loop`): :func:`agm.agl.repl.console.run_console` (prompt_toolkit) or
:func:`agm.agl.repl.plain_console.run_plain_console` (plain line I/O, for a
pipe, comint buffer, or any other non-terminal consumer). The front end is
chosen by :func:`~agm.agl.repl.plain_console.plain_mode_engaged` (non-tty
stdin/stdout, or ``TERM=dumb``) or by the explicit ``--plain`` flag; there is
no flag to force prompt_toolkit onto a pipe. All the interactive logic lives
in :mod:`agm.agl.repl`.

Each REPL entry and its loaded library modules receive ``std/prelude``
glob imports by default. An explicit import whose expansion includes
``std/prelude`` supplies that contribution instead, so plain ``import std/prelude``
leaves prelude names qualified-only. ``--no-stdlib`` disables the automatic import
throughout every loaded REPL program. Imports are qualified by default; tails
and ``use`` declarations opt into bare names.
"""

from __future__ import annotations

import os
import sys

from agm.agent.session import create_agl_session_host
from agm.agl.diagnostics import format_diagnostic
from agm.agl.repl import ReplSession
from agm.agl.repl.plain_console import plain_mode_engaged
from agm.agl.runtime.agents import value_driven_agent_factory
from agm.agl.runtime.host_settings import HostSettingsPolicy
from agm.cli_support.args import ReplArgs
from agm.cli_support.engine_seeds import build_host_engine_seeds
from agm.cli_support.param_config import resolve_module_param_values
from agm.config.context import current_config_context
from agm.config.general import (
    agm_home_dir,
    exec_config_from_merged,
    load_general_config,
    load_repl_config,
    save_repl_setting,
)
from agm.config.module_roots import (
    StdlibResolutionError,
    load_module_roots,
    resolve_lib_root,
    resolve_stdlib_root,
)
from agm.core import dry_run
from agm.core.cleanup import preserve_primary_error
from agm.core.log import (
    LiveTracePathResolver,
    prepare_trace_log_from_decision,
    resolve_trace_decision,
)
from agm.core.toml import toml_dict
from agm.packages.activation import select_package_roots
from agm.packages.development import discover_development_packages


def run(args: ReplArgs) -> None:
    """Run the ``agm repl`` command."""
    ctx = current_config_context()
    try:
        general_config = load_general_config(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)
        merged_config = general_config.merged
        config = exec_config_from_merged(merged_config)
    except ValueError as exc:
        print(f"Error: invalid exec configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    repl_config = load_repl_config(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)

    strict_json = args.strict_json if args.strict_json is not None else config.strict_json
    # Resolve max call depth: CLI > [exec] config (config pragmas are not applied
    # in the REPL).  ``None`` lets the session apply the canonical default.
    call_depth_limit = (
        args.max_call_depth if args.max_call_depth is not None else config.max_call_depth
    )

    # Resolve the CLI > config trace decision ONCE: it both drives the trace
    # file prepared here and seeds the readable ``trace``/``trace-file``
    # registers below, exactly as ``agm exec`` does.
    trace_decision = resolve_trace_decision(
        cli_no_trace=args.no_trace,
        cli_trace=args.trace,
        cli_trace_file=args.trace_file,
        config_trace=config.trace,
        config_trace_file=config.trace_file,
    )

    # Resolve and validate the trace log file up front so an unwritable
    # ``--trace-file`` exits 1 BEFORE the loop starts rather than crashing
    # mid-session.  ``--dry-run`` is side-effect-free (no eval, no trace),
    # mirroring ``agm exec``.
    trace_path = (
        None
        if dry_run.enabled()
        else prepare_trace_log_from_decision(trace_decision, command_name="repl")
    )

    runner_agent = value_driven_agent_factory(idle_timeout=config.timeout, context=ctx)

    session_host = create_agl_session_host(idle_timeout=config.timeout)

    host_settings_policy = HostSettingsPolicy(
        resolve_trace_path=LiveTracePathResolver(command_name="repl", auto_path=trace_path),
    )

    mod_roots_cfg = load_module_roots(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)
    try:
        stdlib_root = resolve_stdlib_root(home=ctx.home, anchor=ctx.cwd)
    except StdlibResolutionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    lib_root = resolve_lib_root(mod_roots_cfg, home=ctx.home)
    try:
        package_roots = select_package_roots(
            home=ctx.home,
            proj_dir=ctx.proj_dir,
            cwd=ctx.cwd,
            development_packages=discover_development_packages(ctx.cwd, home=ctx.home),
        )
    except ValueError as exc:
        print(f"Error: invalid package roots: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # Seed only explicit CLI/config controls.  Trace-service fallbacks remain
    # absent so a ``builtin var`` initializer can provide the setting default.
    # The raw timeout preserves its configured spelling.
    cli_values: dict[str, object | None] = {}
    if args.strict_json is not None:
        cli_values["strict-json"] = args.strict_json
    if args.no_trace:
        cli_values["trace"] = False
    elif args.trace:
        cli_values["trace"] = True
    if args.trace_file is not None:
        cli_values["trace-file"] = args.trace_file
    if args.default_agent is not None:
        cli_values["default-agent"] = args.default_agent
    if args.default_sandbox is not None:
        cli_values["default-sandbox"] = args.default_sandbox

    engine_seeds = build_host_engine_seeds(
        config=config,
        primary_table=toml_dict(merged_config.get("exec")),
        cli_values=cli_values,
    ).merged()

    process_environment = dict(os.environ)
    with preserve_primary_error(session_host.close_all, label="agent session cleanup"):
        session = ReplSession(
            default_strict_json=strict_json,
            default_call_depth_limit=call_depth_limit,
            agent_dispatcher=runner_agent,
            session_host=session_host,
            shell_exec_timeout=config.timeout,
            trace_path=trace_path,
            engine_base=engine_seeds,
            process_environment=process_environment,
            host_settings_policy=host_settings_policy,
            cwd=ctx.cwd,
            stdlib_root=stdlib_root,
            lib_root=lib_root,
            configured_roots=mod_roots_cfg.extra,
            package_roots=package_roots,
            default_stdlib=not args.no_stdlib,
            param_seed_resolver=lambda _module, params: resolve_module_param_values(
                general_config, params
            ),
        )

        # Load and check the session's initial library image now, so any
        # startup failure loading the standard library exits before the
        # console opens and prints its banner, rather than surfacing only
        # once the first entry runs.
        open_diagnostics = session.open()
        if open_diagnostics:
            for diagnostic in open_diagnostics:
                print(f"Error: {format_diagnostic(diagnostic)}", file=sys.stderr)
            raise SystemExit(1)

        history_path = agm_home_dir(home=ctx.home) / "repl_history"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        def on_setting_save(key: str, value: str | bool) -> None:
            save_repl_setting(key, value, home=ctx.home)

        # ``--quiet`` disables echo for this session only and never persists,
        # overriding a saved ``[repl] echo = true``; absent ``--quiet`` the
        # persisted (or default) ``[repl] echo`` applies.
        echo = repl_config.echo and not args.quiet

        # ``--dry-run`` means type-check only in the REPL: every entry runs the full
        # static pipeline but is never evaluated, so no agent/exec calls fire and no
        # bindings are persisted.  It reads the same global flag ``agm exec`` honours.
        #
        # The front end is chosen once, here: ``--plain`` forces the plain line
        # front end; otherwise ``plain_mode_engaged`` auto-detects it from
        # stdin/stdout (a pipe, redirected file, or a dumb terminal). There is no
        # flag to force prompt_toolkit onto a non-terminal. The ``console`` import
        # stays local so a plain session never pulls in prompt_toolkit; the
        # ``plain_console`` import is local too for symmetry and late binding —
        # ``plain_mode_engaged`` is already imported from it at module top, so
        # this local import defers nothing on the plain branch, but the late
        # binding is what test fixtures rely on when they monkeypatch it.
        if args.plain or plain_mode_engaged(stdin=sys.stdin, stdout=sys.stdout, env=os.environ):
            from agm.agl.repl.plain_console import run_plain_console

            run_plain_console(
                session,
                echo=echo,
                echo_unit=repl_config.echo_unit,
                check_only=dry_run.enabled(),
                theme=repl_config.theme,
                on_setting_save=on_setting_save,
                stdin=sys.stdin,
                stdout=sys.stdout,
            )
            return

        from agm.agl.repl.console import run_console

        run_console(
            session,
            echo=echo,
            echo_unit=repl_config.echo_unit,
            check_only=dry_run.enabled(),
            history_path=history_path,
            theme=repl_config.theme,
            on_setting_save=on_setting_save,
        )
