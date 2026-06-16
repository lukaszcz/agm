"""Implementation of the ``agm exec FILE`` command.

Behaviour: read the ``.agl`` source — either from the inline ``-c/--command``
argument or from the source file (exit 1 if unreadable), load the
``[exec]`` configuration, construct a ``WorkflowRuntime`` with the resolved
settings, call ``runtime.run`` (or a static-only dry run under ``--dry-run``),
print diagnostics to stderr, and exit per the exit-code contract.

Warnings (``result.warnings``) and error diagnostics (``result.diagnostics``)
are two separate channels: warnings are printed to stderr like errors but never
affect the exit code; only error-severity diagnostics yield exit 1.  Warnings
carry a ``warning:`` prefix (``warning: line N: message``) to disambiguate them
from errors on the shared stderr channel.

Exit-code contract (plan §10.1):
    0  success (or a clean ``--dry-run`` static check)
    1  pre-execution failure (unreadable file, static errors, param validation)
    2  program executed but ended with an uncaught AgL exception

Flag notes:
    - ``--strict-json`` controls JSON-codec strictness: when set, agents must
      return exactly one bare JSON value; the default is lenient recovery
      (fence/prose stripping + trivial repair, then strict schema validation).
      A source-level ``strict_json`` call option overrides this default.
    - ``--log-file PATH`` writes a structured JSONL trace under the given path.
    - ``--no-log`` disables trace logging entirely.
    - ``--runner COMMAND`` overrides the default agent runner command from config.
      When set, it is used as the default runner for all unnamed agents.
    - ``--dry-run`` (global flag) runs only the static pipeline + contract
      materialization and never writes a trace (side-effect-free).
    - Each ``param`` declaration in the source program becomes a ``--<name>``
      option.  Bool params use the ``--name/--no-name`` flag form.  Collision
      with built-in exec options is detected eagerly and reported before execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agm.agent.runner import split_command
from agm.agl import WorkflowRuntime
from agm.agl.runtime.agents import runner_backed_agent_factory
from agm.commands.agent_io import default_agent_runner
from agm.commands.args import ExecArgs
from agm.commands.param_options import (
    check_param_collisions,
    parse_param_tokens,
    resolve_param_values,
)
from agm.config.context import current_config_context
from agm.config.general import load_exec_config, load_params_config
from agm.core import dry_run
from agm.core.fs import read_text_arg
from agm.core.log import prepare_trace_log
from agm.parser import exit_with_usage_error


def run(args: ExecArgs) -> None:
    """Run the ``agm exec`` command."""
    # The program source comes either from an inline ``-c/--command`` argument
    # or from a file.  The CLI layer guarantees exactly one is provided; the
    # defensive ``else`` keeps ``run`` safe when called directly.
    if args.command is not None:
        source = args.command
    elif args.file is not None:
        source = read_text_arg(Path(args.file))
    else:
        print("Error: exec requires either a FILE or -c/--command", file=sys.stderr)
        raise SystemExit(1)

    # Load [exec] configuration; CLI flags override config values.
    ctx = current_config_context()
    try:
        config = load_exec_config(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)
    except ValueError as exc:
        print(f"Error: invalid exec configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    strict_json = args.strict_json if args.strict_json is not None else config.strict_json
    loop_limit = args.max_iters if args.max_iters is not None else config.default_loop_limit

    # Resolve + validate the trace log file up front (F2a/F6).  --dry-run is
    # side-effect-free: no trace is written regardless of --log-file (plan §10.1).
    if dry_run.enabled():
        log_file = None
    else:
        log_file = prepare_trace_log(
            command_name="exec", no_log=args.no_log, log_file=args.log_file
        )

    # ----------------------------------------------------------------
    # Resolve the runner command: CLI flag > [exec] config > shared loop
    # default (the same default used by agm loop/review, per plan §9.5).
    # ----------------------------------------------------------------
    runner_cmd = args.runner or config.runner or default_agent_runner()

    # Validate the resolved runner command eagerly: malformed quoting (e.g.
    # unclosed quote) and whitespace-only values are caught here via
    # split_command, which also handles the ValueError from shlex.split for
    # malformed quoting.  This honours the exit-1 = pre-execution contract
    # (plan §10.1) and deduplicates the logic already in split_command.
    split_command(runner_cmd, kind="runner")

    # ----------------------------------------------------------------
    # Resolve declared agents and wire each one explicitly (plan §9).
    #
    # The source program OWNS the agent name set: every named agent must be
    # declared.  We register each DECLARED agent against a single runner-backed
    # factory whose per-agent command map merges, in precedence order
    # (high → low; decision §4):
    #
    #     [exec.agents.<name>]   (config, per-agent)
    #     source `agent` runner hint
    #     resolved default runner (runner_cmd, the floor)
    #
    # ``prepare`` parses + scopes the source ONCE (independent of registrations);
    # the same ``PreparedProgram`` is handed to ``run_prepared`` below, so the
    # program is never parsed or scoped twice.  On a source with parse/scope
    # errors ``declared_agents`` is ``()`` and ``run_prepared`` resurfaces the
    # captured diagnostic (exit 1).
    prepared = WorkflowRuntime.prepare(source)
    decls = prepared.declared_agents
    source_hints = {d.name: d.runner for d in decls if d.runner is not None}
    # Config wins over source hints (dict merge: later keys override earlier).
    per_agent_cmds = {**source_hints, **config.agents}

    # Validate each DECLARED agent's resolved runner command eagerly, honouring
    # the same pre-execution contract as the default runner above: a malformed
    # (e.g. unclosed quote) or empty per-agent command — from a source `agent`
    # runner hint or an `[exec.agents.<name>]` config entry — exits 1 before any
    # statement runs, rather than failing lazily mid-execution at dispatch.
    # Only declared (dispatchable) agents are checked; a config entry for an
    # agent the program never declares is inert and never validated.
    for d in decls:
        cmd = per_agent_cmds.get(d.name)
        if cmd is not None:
            split_command(cmd, kind="runner")

    # One factory backs ``prompt`` (the default) and every declared name; it
    # dispatches by ``request.agent`` against ``per_agent_cmds``, falling back
    # to the default runner (the floor).  ``command_with_prompt_target``
    # substitutes ``%%`` / ``%{PROMPT_FILE}`` for source hints and config
    # commands alike.
    factory = runner_backed_agent_factory(
        default_runner_cmd=runner_cmd,
        per_agent_cmds=per_agent_cmds,
        idle_timeout=config.timeout,
    )

    runtime = WorkflowRuntime(
        default_loop_limit=loop_limit,
        default_strict_json=strict_json,
        default_agent=factory,
        shell_exec_timeout=config.timeout,
    )

    # Register every declared agent so the registered set equals the declared
    # set: M4 reconciliation always passes; config-only agents the source never
    # declares stay inert (NOT registered), per plan §9.
    for d in decls:
        runtime.register_agent(d.name, factory)

    # ----------------------------------------------------------------
    # Discover param declarations and validate CLI tokens.
    #
    # Only runs when the front-end (parse/scope) succeeded (prepared.resolved
    # is not None) AND typecheck succeeded (discovery.checked is not None).
    # On front-end failure we skip CLI-param validation and hand the prepared
    # program directly to run_prepared, which resurfaces the captured diagnostic.
    # ----------------------------------------------------------------
    discovery = runtime.discover_params(prepared)

    # Surface discovery warnings on stderr (e.g. non-exhaustive case at
    # typecheck time).  They never affect the exit code.
    for diag in discovery.warnings:
        print(f"warning: line {diag.line}: {diag.message}", file=sys.stderr)

    # ----------------------------------------------------------------
    # Resolve param values from config + CLI (D2), ONLY when the front-end
    # and typecheck both succeeded.  When ``discovery.checked is None`` the
    # param set is unknown (parse/scope/typecheck failed), so we skip the
    # whole param layer — no config lookup, no collision/token validation, no
    # undeclared-key warnings — and hand the prepared program straight to
    # run_prepared, which resurfaces the captured diagnostic (exit 1).  Doing
    # otherwise would spew misleading "undeclared config key" warnings that
    # mask the real parse/type error.
    # ----------------------------------------------------------------
    external_inputs: dict[str, object] = {}
    checked = discovery.checked  # None when front-end or typecheck failed

    if checked is not None:
        # Detect param names that collide with built-in exec flags before
        # parsing tokens: a collision is a program error (rename the param).
        collision_errors = check_param_collisions(discovery.params)
        if collision_errors:
            for err in collision_errors:
                print(f"Error: {err}", file=sys.stderr)
            raise SystemExit(1)

        # Parse the leftover ``ctx.args`` tokens into the param dict.
        try:
            cli_params = parse_param_tokens(discovery.params, args.param_tokens)
        except ValueError as exc:
            exit_with_usage_error(["exec"], f"error: {exc}")

        # Resolve the program key for [params.<key>] config lookup (D2).
        # Priority: declared ``program NAME`` > .agl file stem > None (inline
        # -c with no program decl has no config table).
        if discovery.program_name is not None:
            param_config_key: str | None = discovery.program_name
        elif args.file is not None:
            param_config_key = Path(args.file).stem
        else:
            param_config_key = None

        if param_config_key is not None:
            config_param_values = load_params_config(
                param_config_key,
                home=ctx.home,
                proj_dir=ctx.proj_dir,
                cwd=ctx.cwd,
            )
        else:
            config_param_values = {}

        # Merge config and CLI values (CLI wins; undeclared config keys warn).
        declared_names = {p.name for p in discovery.params}
        external_inputs, config_warnings = resolve_param_values(
            declared_names,
            config_param_values,
            cli_params,
            program_name=param_config_key,
        )
        for msg in config_warnings:
            print(msg, file=sys.stderr)

    # ``run_prepared`` accepts a ``Mapping[str, object]``.  Reuse the
    # ``PreparedProgram`` from above — no second parse/scope of the source.
    # Pass ``checked`` to skip a redundant typecheck inside run_prepared
    # (it already ran during discover_params when checked is not None).
    result = runtime.run_prepared(
        prepared,
        inputs=external_inputs,
        check_only=dry_run.enabled(),
        log_file=log_file,
        checked=checked,
    )

    # Warnings live on their own channel and never affect the exit code;
    # ``result.diagnostics`` holds only error-severity pre-execution failures.
    # Warnings carry a ``warning:`` prefix to disambiguate them from error
    # diagnostics on the shared stderr channel (F8).
    for diag in result.warnings:
        print(f"warning: line {diag.line}: {diag.message}", file=sys.stderr)

    if result.ok:
        # Print the static call-site inventory when running under --dry-run.
        if dry_run.enabled() and result.call_sites:
            print("call-sites:")
            for site in result.call_sites:
                schema_tag = ", schema: yes" if site.has_schema else ""
                policy_tag = (
                    f", policy: {site.parse_policy}" if site.parse_policy != "default" else ""
                )
                print(
                    f"  line {site.line}:{site.col}: {site.callee} "
                    f"→ {site.target_type} "
                    f"[{site.codec_name}{schema_tag}{policy_tag}]"
                )
        return

    # Pre-execution failure: print error diagnostics and exit 1.
    if result.error is None:
        for diag in result.diagnostics:
            print(f"line {diag.line}: {diag.message}", file=sys.stderr)
        raise SystemExit(1)

    # Uncaught AgL exception: print and exit 2 (design §12.6: include source
    # location and trace_id in the error line so the caller can correlate the
    # error with the trace file and the source program).
    print(result.error.to_message(include_trace_id=True), file=sys.stderr)
    raise SystemExit(2)
