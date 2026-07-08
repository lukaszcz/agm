"""Implementation of the ``agm repl`` command.

Launches an interactive read-eval-print loop for the AgL workflow language.
The REPL shares ``agm exec``'s ``[exec]`` configuration (runner / agents /
timeout), so an interactive session evaluates entries with the same agent
backing a batch ``agm exec`` run would use.

The command itself is thin: it resolves configuration the same way ``exec``
does, builds a runner-backed agent wrapped in a confirming wrapper, constructs a
:class:`ReplSession`, and hands control to
:func:`agm.agl.repl.console.run_console`.  All the interactive logic lives in
:mod:`agm.agl.repl`.

Agent calls are gated: a single shared :class:`AgentMode` (``confirm`` by
default, ``auto``; ``confirm`` under ``--confirm-agents``) is passed to BOTH the confirming
wrapper and the console, so the ``:agent`` meta-command, an ``always`` answer,
and the wrapper all stay in sync.  Trace logging (``--log-file`` / ``--no-log``)
and param config loading are wired from the same config stack as ``agm exec``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agm.agent.config import default_agent_runner
from agm.agent.runner import split_command
from agm.agl.repl import ReplSession
from agm.agl.repl.agentmode import AgentMode
from agm.agl.repl.agents import ConfirmingAgent
from agm.agl.runtime.agents import runner_backed_agent_factory
from agm.agl.runtime.params import build_engine_config_base, raw_option_str
from agm.agl.semantics.values import Value
from agm.cli_support.args import ReplArgs
from agm.config.context import current_config_context
from agm.config.general import (
    agm_home_dir,
    exec_config_from_merged,
    load_merged_config,
    load_program_config,
    load_repl_config,
    save_repl_theme,
)
from agm.config.module_roots import load_module_roots, resolve_lib_root, resolve_stdlib_root
from agm.core import dry_run
from agm.core.log import prepare_trace_log_from_layers
from agm.core.toml import toml_dict


def run(args: ReplArgs) -> None:
    """Run the ``agm repl`` command."""
    ctx = current_config_context()
    try:
        merged_config = load_merged_config(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)
        config = exec_config_from_merged(merged_config)
    except ValueError as exc:
        print(f"Error: invalid exec configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    repl_config = load_repl_config(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)

    strict_json = args.strict_json if args.strict_json is not None else config.strict_json
    loop_limit = args.max_iters if args.max_iters is not None else config.default_loop_limit
    # Resolve max call depth: CLI > [exec] config (config pragmas are not applied
    # in the REPL).  ``None`` lets the session apply the canonical default.
    call_depth_limit = (
        args.max_call_depth if args.max_call_depth is not None else config.max_call_depth
    )

    # Resolve the runner command: CLI flag > [exec] config > shared default,
    # exactly as ``agm exec`` does (the REPL shares the exec agent backing).
    runner_cmd = args.runner or config.runner or default_agent_runner()
    # Validate the resolved runner eagerly (malformed quoting / empty value
    # surface here as a clean error before the loop starts).
    split_command(runner_cmd, kind="runner")

    # Resolve and validate the trace log file.  ``--dry-run`` is side-effect-free
    # (no eval, no trace), mirroring ``agm exec``.
    trace_path = _resolve_trace_path(args, config_log=config.log, config_log_file=config.log_file)

    runner_agent = runner_backed_agent_factory(
        default_runner_cmd=runner_cmd,
        per_agent_cmds=config.agents,
        idle_timeout=config.timeout,
    )

    # ONE shared agent-mode holder: passed to BOTH the confirming wrapper and the
    # console, so ``:agent``/``always`` and the wrapper observe the same mode.
    # ``--confirm-agents`` starts in ``confirm``; otherwise auto (decision 2).
    agent_mode = AgentMode(mode="confirm" if args.confirm_agents else "auto")

    # Importing the console pulls in prompt_toolkit; defer it so non-interactive
    # code paths never pay for the terminal dependency.
    from agm.agl.repl.console import make_console_confirm, run_console

    confirming_agent = ConfirmingAgent(
        runner_agent, agent_mode, confirm=make_console_confirm()
    )

    def _params_config_loader(program_name: str) -> dict[str, object]:
        return load_program_config(
            program_name, home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd
        )

    mod_roots_cfg = load_module_roots(
        home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd
    )
    stdlib_root = resolve_stdlib_root(home=ctx.home)
    lib_root = resolve_lib_root(mod_roots_cfg, home=ctx.home)

    # Build the [exec] engine base for the four start-resolved engine keys
    # (runner, log, log-file, timeout) that the session uses as the config_base
    # floor for bare ``config KEY`` bindings.  strict-json and max-iters are
    # tracked by the session's own persisted settings and override these values
    # in _build_config_base; they are included here so the dict is complete for
    # any consumer that reads all six keys.
    #
    # The raw timeout string is read from the TOML table via raw_option_str so
    # the binding holds the original written value (e.g. "30s") rather than a
    # parsed float (e.g. "30.0").
    exec_raw_table = toml_dict(merged_config.get("exec"))
    raw_timeout_str = raw_option_str(exec_raw_table, {}, "timeout")
    # Omit max-iters when unset (None = valve OFF) so build_engine_config_base
    # falls back to its _ENGINE_DEFAULTS entry (the bare-config floor).
    repl_base_raw: dict[str, object] = {
        "strict-json": strict_json,
        "runner": runner_cmd,
        "log": config.log,
        "timeout": raw_timeout_str,
        "log-file": config.log_file,
    }
    if loop_limit is not None:
        repl_base_raw["max-iters"] = loop_limit
    engine_base: dict[str, Value] = build_engine_config_base(repl_base_raw)

    session = ReplSession(
        default_strict_json=strict_json,
        default_loop_limit=loop_limit,
        default_call_depth_limit=call_depth_limit,
        default_agent=confirming_agent,
        shell_exec_timeout=config.timeout,
        trace_path=trace_path,
        params_config_loader=_params_config_loader,
        engine_base=engine_base,
        cwd=ctx.cwd,
        stdlib_root=stdlib_root,
        lib_root=lib_root,
        configured_roots=mod_roots_cfg.extra,
    )

    history_path = agm_home_dir(home=ctx.home) / "repl_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # ``--dry-run`` means type-check only in the REPL: every entry runs the full
    # static pipeline but is never evaluated, so no agent/exec calls fire and no
    # bindings are persisted.  It reads the same global flag ``agm exec`` honours.
    run_console(
        session,
        echo=not args.quiet,
        check_only=dry_run.enabled(),
        agent_mode=agent_mode,
        history_path=history_path,
        theme=repl_config.theme,
        on_theme_save=lambda t: save_repl_theme(t, home=ctx.home),
    )


def _resolve_trace_path(
    args: ReplArgs, config_log: bool, config_log_file: str | None
) -> Path | None:
    """Resolve + validate the JSONL trace path, or ``None`` (dry-run / disabled).

    Mirrors ``agm exec`` via the shared :func:`prepare_trace_log_from_layers`:
    ``--dry-run`` writes no trace; otherwise the path is resolved and validated
    up front so an unwritable ``--log-file`` exits 1 BEFORE the loop starts
    rather than crashing mid-session.
    """
    if dry_run.enabled():
        return None
    return prepare_trace_log_from_layers(
        command_name="repl",
        cli_no_log=args.no_log,
        cli_log=args.log,
        cli_log_file=args.log_file,
        config_log=config_log,
        config_log_file=config_log_file,
    )
