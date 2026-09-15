"""Implementation of the ``agm repl`` command.

Launches an interactive read-eval-print loop for the AgL workflow language.
The REPL shares ``agm exec``'s ``[exec]`` configuration (default-agent and
timeout), so an interactive session evaluates entries with the same agent
dispatch backing a batch ``agm exec`` run would use.

The command itself is thin: it resolves configuration the same way ``exec``
does, builds a value-driven dispatcher wrapped in a confirming wrapper,
constructs a :class:`ReplSession`, then hands control to one of two front
ends sharing the same UI-free loop core (:mod:`agm.agl.repl.loop`):
:func:`agm.agl.repl.console.run_console` (prompt_toolkit) or
:func:`agm.agl.repl.plain_console.run_plain_console` (plain line I/O, for a
pipe, comint buffer, or any other non-terminal consumer). The front end is
chosen by :func:`~agm.agl.repl.plain_console.plain_mode_engaged` (non-tty
stdin/stdout, or ``TERM=dumb``) or by the explicit ``--plain`` flag; there is
no flag to force prompt_toolkit onto a pipe. All the interactive logic lives
in :mod:`agm.agl.repl`.

Agent calls are gated: a single shared :class:`AgentMode` (``confirm`` by
default, ``auto``; ``confirm`` under ``--confirm-agents``) is passed to BOTH the confirming
wrapper and the chosen front end, so the ``:agent`` meta-command, an ``always``
answer, and the wrapper all stay in sync.  Trace logging (``--log-file`` /
``--no-log``) Each REPL entry and its loaded library modules receive ``std/prelude``
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
from agm.agl.repl.agentmode import AgentMode
from agm.agl.repl.agents import ConfirmingAgent
from agm.agl.repl.loop import make_console_confirm
from agm.agl.repl.plain_console import plain_mode_engaged
from agm.agl.runtime.agents import value_driven_agent_factory
from agm.agl.runtime.host_settings import HostSettingsPolicy
from agm.cli_support.args import ReplArgs
from agm.cli_support.engine_seeds import build_host_engine_seeds, check_max_iters
from agm.cli_support.param_config import resolve_module_param_values
from agm.config.context import current_config_context
from agm.config.general import (
    agm_home_dir,
    exec_config_from_merged,
    load_general_config,
    load_repl_config,
    save_repl_theme,
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
    resolve_log_decision,
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
    check_max_iters(args.max_iters)
    loop_limit = args.max_iters if args.max_iters is not None else config.default_loop_limit
    # Resolve max call depth: CLI > [exec] config (config pragmas are not applied
    # in the REPL).  ``None`` lets the session apply the canonical default.
    call_depth_limit = (
        args.max_call_depth if args.max_call_depth is not None else config.max_call_depth
    )

    # Resolve the CLI > config logging decision ONCE: it both drives the trace
    # file prepared here and seeds the readable ``log``/``log-file`` registers
    # below, exactly as ``agm exec`` does.
    log_decision = resolve_log_decision(
        cli_no_log=args.no_log,
        cli_log=args.log,
        cli_log_file=args.log_file,
        config_log=config.log,
        config_log_file=config.log_file,
    )

    # Resolve and validate the trace log file up front so an unwritable
    # ``--log-file`` exits 1 BEFORE the loop starts rather than crashing
    # mid-session.  ``--dry-run`` is side-effect-free (no eval, no trace),
    # mirroring ``agm exec``.
    trace_path = (
        None
        if dry_run.enabled()
        else prepare_trace_log_from_decision(log_decision, command_name="repl")
    )

    runner_agent = value_driven_agent_factory(idle_timeout=config.timeout)

    # ONE shared agent-mode holder: passed to BOTH the confirming wrapper and the
    # chosen front end, so ``:agent``/``always`` and the wrapper observe the same
    # mode.  ``--confirm-agents`` starts in ``confirm``; otherwise auto (decision 2).
    agent_mode = AgentMode(mode="confirm" if args.confirm_agents else "auto")

    # UI-free: shared by both front ends, so building it never pulls in
    # prompt_toolkit even when the plain front end is the one that runs.
    confirm_agent_call = make_console_confirm()
    confirming_agent = ConfirmingAgent(runner_agent, agent_mode, confirm=confirm_agent_call)

    session_host = create_agl_session_host(
        idle_timeout=config.timeout,
        confirm_session=confirming_agent.confirm_session_values,
    )

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
    # The raw timeout preserves its configured spelling. A host Agent value
    # (``--default-agent``/``[exec] default-agent``) becomes an override spliced into
    # the session's own first-loaded ``std/config`` rather than a seed value.
    cli_values: dict[str, object | None] = {}
    if args.strict_json is not None:
        cli_values["strict-json"] = args.strict_json
    if args.max_iters is not None:
        cli_values["max-iters"] = args.max_iters
    if args.no_log:
        cli_values["log"] = False
    elif args.log:
        cli_values["log"] = True
    if args.log_file is not None:
        cli_values["log-file"] = args.log_file

    engine_seeds = build_host_engine_seeds(
        config=config,
        primary_table=toml_dict(merged_config.get("exec")),
        cli_values=cli_values,
        default_agent=args.default_agent,
    )

    process_environment = dict(os.environ)
    with preserve_primary_error(session_host.close_all, label="agent session cleanup"):
        session = ReplSession(
            default_strict_json=strict_json,
            default_loop_limit=loop_limit,
            default_call_depth_limit=call_depth_limit,
            agent_dispatcher=confirming_agent,
            session_host=session_host,
            shell_exec_timeout=config.timeout,
            trace_path=trace_path,
            engine_base=engine_seeds.values,
            process_environment=process_environment,
            setting_overrides=engine_seeds.overrides,
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

        # Load and check the session's initial library image now, so a rejected
        # ``--default-agent``/``[exec] default-agent`` override (or any other startup
        # failure loading the standard library) exits before the console opens
        # and prints its banner, rather than surfacing only once the first entry
        # happens to load ``std/config``.
        open_diagnostics = session.open()
        if open_diagnostics:
            for diagnostic in open_diagnostics:
                print(f"Error: {format_diagnostic(diagnostic)}", file=sys.stderr)
            raise SystemExit(1)

        history_path = agm_home_dir(home=ctx.home) / "repl_history"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        def on_theme_save(new_theme: str) -> None:
            save_repl_theme(new_theme, home=ctx.home)

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
                echo=not args.quiet,
                check_only=dry_run.enabled(),
                agent_mode=agent_mode,
                theme=repl_config.theme,
                on_theme_save=on_theme_save,
                stdin=sys.stdin,
                stdout=sys.stdout,
            )
            return

        from agm.agl.repl.console import run_console

        run_console(
            session,
            echo=not args.quiet,
            check_only=dry_run.enabled(),
            agent_mode=agent_mode,
            history_path=history_path,
            theme=repl_config.theme,
            on_theme_save=on_theme_save,
        )
