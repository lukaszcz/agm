"""Implementation of the ``agm repl`` command.

Launches an interactive read-eval-print loop for the AgL workflow language.
The REPL shares ``agm exec``'s ``[exec]`` configuration (runner / agents /
timeout), so an interactive session evaluates entries with the same agent
backing a batch ``agm exec`` run would use.

The command itself is thin: it resolves configuration the same way ``exec``
does, builds a runner-backed agent, constructs a :class:`ReplSession`, and hands
control to :func:`agm.agl.repl.console.run_console`.  All the interactive logic
lives in :mod:`agm.agl.repl`.

Milestone note: M2 wires the runnable end-to-end loop.  Agent-call confirmation
modes (``--auto-agents`` / ``:agent``), trace logging (``--no-log`` /
``--log-file``), and ``--input`` pre-seed are deferred to M4 — each deferral is
marked with an ``# M4:`` comment so the later milestone is a localized change
with no signature churn.
"""

from __future__ import annotations

import sys

from agm.agent.runner import split_command
from agm.agl.repl import ReplSession
from agm.agl.runtime.agents import runner_backed_agent_factory
from agm.commands.agent_io import default_agent_runner
from agm.commands.args import ReplArgs
from agm.config.context import current_config_context
from agm.config.general import load_exec_config
from agm.core import dry_run


def run(args: ReplArgs) -> None:
    """Run the ``agm repl`` command."""
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

    # M4: wrap this agent in the confirming wrapper honouring ``--auto-agents``
    # / the ``:agent`` mode before constructing the session.
    runner_agent = runner_backed_agent_factory(
        default_runner_cmd=runner_cmd,
        per_agent_cmds=config.agents,
        idle_timeout=config.timeout,
    )

    session = ReplSession(
        default_loop_limit=loop_limit,
        default_strict_json=strict_json,
        default_agent=runner_agent,
        shell_exec_timeout=config.timeout,
    )

    # M4: pre-seed declared inputs from ``args.inputs`` (``--input KEY=VALUE``)
    # via ``session.set_input`` once the input flow lands.

    # M4: thread trace logging (``args.no_log`` / ``args.log_file`` resolved via
    # ``resolve_log_file``) into ``eval_entry`` once it accepts a trace path.

    history_path = ctx.home / "repl_history"

    # Importing the console pulls in prompt_toolkit; defer it so non-interactive
    # code paths never pay for the terminal dependency.
    from agm.agl.repl.console import run_console

    # ``--dry-run`` means type-check only in the REPL: every entry runs the full
    # static pipeline but is never evaluated, so no agent/exec calls fire and no
    # bindings are persisted.  It reads the same global flag ``agm exec`` honours.
    run_console(
        session,
        echo=not args.quiet,
        check_only=dry_run.enabled(),
        history_path=history_path,
    )
