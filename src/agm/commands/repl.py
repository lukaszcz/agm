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
default, ``auto`` under ``--auto-agents``) is passed to BOTH the confirming
wrapper and the console, so the ``:agent`` meta-command, an ``always`` answer,
and the wrapper all stay in sync.  Trace logging (``--log-file`` / ``--no-log``)
and ``--input KEY=VALUE`` pre-seed are wired exactly as ``agm exec`` resolves
them.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agm.agent.runner import split_command
from agm.agl.repl import ReplSession
from agm.agl.repl.agentmode import AgentMode
from agm.agl.repl.agents import ConfirmingAgent
from agm.agl.runtime.agents import runner_backed_agent_factory
from agm.commands.agent_io import default_agent_runner
from agm.commands.args import ReplArgs
from agm.config.context import current_config_context
from agm.config.general import load_exec_config
from agm.core import dry_run
from agm.core.cli_helpers import parse_inputs
from agm.core.fs import append_text, mkdir
from agm.core.log import resolve_log_file


def run(args: ReplArgs) -> None:
    """Run the ``agm repl`` command."""
    # Parse --input k=v pairs up front (malformed pairs exit 1, like exec).
    try:
        inputs = parse_inputs(args.inputs)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    ctx = current_config_context()
    try:
        config = load_exec_config(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)
    except ValueError as exc:
        print(f"Error: invalid exec configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    strict_json = args.strict_json if args.strict_json is not None else config.strict_json
    loop_limit = args.max_iters if args.max_iters is not None else config.default_loop_limit

    # Resolve the runner command: CLI flag > [exec] config > shared default,
    # exactly as ``agm exec`` does (the REPL shares the exec agent backing).
    runner_cmd = args.runner or config.runner or default_agent_runner()
    # Validate the resolved runner eagerly (malformed quoting / empty value
    # surface here as a clean error before the loop starts).
    split_command(runner_cmd, kind="runner")

    # Resolve and validate the trace log file.  ``--dry-run`` is side-effect-free
    # (no eval, no trace), mirroring ``agm exec``.
    trace_path = _resolve_trace_path(args)

    runner_agent = runner_backed_agent_factory(
        default_runner_cmd=runner_cmd,
        per_agent_cmds=config.agents,
        idle_timeout=config.timeout,
    )

    # ONE shared agent-mode holder: passed to BOTH the confirming wrapper and the
    # console, so ``:agent``/``always`` and the wrapper observe the same mode.
    # ``--auto-agents`` starts in ``auto``; otherwise confirm-each-call (decision 2).
    agent_mode = AgentMode(mode="auto" if args.auto_agents else "confirm")

    # Importing the console pulls in prompt_toolkit; defer it so non-interactive
    # code paths never pay for the terminal dependency.
    from agm.agl.repl.console import make_console_confirm, run_console

    confirming_agent = ConfirmingAgent(
        runner_agent, agent_mode, confirm=make_console_confirm()
    )

    session = ReplSession(
        default_loop_limit=loop_limit,
        default_strict_json=strict_json,
        default_agent=confirming_agent,
        shell_exec_timeout=config.timeout,
        trace_path=trace_path,
    )

    # Pre-seed declared inputs from ``--input KEY=VALUE`` before the loop starts.
    for key, value in inputs.items():
        session.preset_input(key, value)

    history_path = ctx.home / "repl_history"

    # ``--dry-run`` means type-check only in the REPL: every entry runs the full
    # static pipeline but is never evaluated, so no agent/exec calls fire and no
    # bindings are persisted.  It reads the same global flag ``agm exec`` honours.
    run_console(
        session,
        echo=not args.quiet,
        check_only=dry_run.enabled(),
        agent_mode=agent_mode,
        history_path=history_path,
    )


def _resolve_trace_path(args: ReplArgs) -> Path | None:
    """Resolve + validate the JSONL trace path, or ``None`` (dry-run / --no-log).

    Mirrors ``agm exec``: ``--dry-run`` writes no trace; otherwise the path is
    resolved via :func:`resolve_log_file` and validated up front (mkdir +
    touch-append) so an unwritable ``--log-file`` exits 1 BEFORE the loop starts
    rather than crashing mid-session.
    """
    if dry_run.enabled():
        return None
    log_file: Path | None = resolve_log_file(
        command_name="repl",
        no_log=args.no_log,
        log_file=args.log_file,
        unique=True,
    )
    if log_file is not None:
        try:
            mkdir(log_file.parent, parents=True, exist_ok=True)
            # Touch-append confirms writability without writing a record, so the
            # first traced entry starts from a clean file.
            append_text(log_file, "", encoding="utf-8")
        except OSError as exc:
            print(f"Error: cannot write trace log to {log_file}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    return log_file
