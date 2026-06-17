"""Implementation of the ``agm exec FILE`` command.

Behaviour: read the ``.agl`` source — either from the inline ``-c/--command``
argument or from the source file (exit 1 if unreadable), load the
``[exec]`` configuration, construct a ``WorkflowRuntime`` with the resolved
settings, call ``runtime.run`` (or a static-only dry run under ``--dry-run``),
print diagnostics to stderr, and exit per the exit-code contract.

Warnings (``result.warnings``) and error diagnostics (``result.diagnostics``)
are two separate channels: warnings are printed to stderr like errors but never
affect the exit code; only error-severity diagnostics yield exit 1.  The
diagnostic severity is included in compiler-style output, e.g.
``path.agl:1:5: warning: message`` or ``1:5: error: message`` for inline
``-c/--command`` source.

Exit-code contract (plan §10.1):
    0  success (or a clean ``--dry-run`` static check)
    1  pre-execution failure (unreadable file, static errors, param validation)
    2  program executed but ended with an uncaught AgL exception

Flag notes:
    - ``--strict-json`` controls JSON-codec strictness: when set, agents must
      return exactly one bare JSON value; the default is lenient recovery
      (fence/prose stripping + trivial repair, then strict schema validation).
      A source-level ``strict_json`` call option overrides this default.
    - Trace logging is OFF by default.  ``--log`` enables it (auto-named path);
      ``--log-file PATH`` writes to PATH; ``--no-log`` disables it.  At most one
      of these three flags may be given (mutually exclusive).  A source-level
      ``config log = true`` pragma or ``[exec] log = true`` in config also enables
      logging; CLI flags override pragmas (CLI > pragma > config).
    - ``--runner COMMAND`` overrides the default agent runner command from config.
      When set, it is used as the default runner for all unnamed agents.
    - Source ``config KEY = VALUE`` pragmas (header-only) override config-file
      settings for ``strict_json``, ``max_iters``, ``runner``, ``timeout``,
      ``log``, and ``log_file``.  CLI flags always take precedence.
    - ``--dry-run`` (global flag) runs only the static pipeline + contract
      materialization and never writes a trace (side-effect-free).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypeVar

from agm.agent.runner import split_command
from agm.agl import WorkflowRuntime
from agm.agl.diagnostics import format_diagnostic
from agm.agl.runtime.agents import runner_backed_agent_factory
from agm.agl.syntax.nodes import PragmaValue
from agm.commands.agent_io import default_agent_runner
from agm.commands.args import ExecArgs
from agm.commands.param_options import (
    check_param_collisions,
    parse_param_tokens,
    resolve_param_values,
)
from agm.config.context import current_config_context
from agm.config.general import (
    load_exec_config,
    load_merged_config,
    params_config_from_merged,
    parse_timeout,
)
from agm.core import dry_run
from agm.core.fs import read_text_arg
from agm.core.log import prepare_trace_log_from_layers
from agm.parser import exit_with_usage_error

_T = TypeVar("_T")


def _first(*values: _T | None) -> _T | None:
    """Return the first non-None value, or None if all are None."""
    return next((v for v in values if v is not None), None)


def _typed_pragma(pragmas: dict[str, PragmaValue], key: str, typ: type[_T]) -> _T | None:
    """Return the pragma value for *key* if present and of type *typ*, else None."""
    value = pragmas.get(key)
    return value if isinstance(value, typ) else None


def run(args: ExecArgs) -> None:
    """Run the ``agm exec`` command."""
    # The program source comes either from an inline ``-c/--command`` argument
    # or from a file.  The CLI layer guarantees exactly one is provided; the
    # defensive ``else`` keeps ``run`` safe when called directly.
    if args.command is not None:
        source = args.command
        diagnostic_source_name: str | None = None
    elif args.file is not None:
        source = read_text_arg(Path(args.file))
        diagnostic_source_name = args.file
    else:
        print("Error: exec requires either a FILE or -c/--command", file=sys.stderr)
        raise SystemExit(1)

    # Load the merged config once; derive [exec] and [params.<program>] from it.
    ctx = current_config_context()
    try:
        merged_config = load_merged_config(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)
        config = load_exec_config(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)
    except ValueError as exc:
        print(f"Error: invalid exec configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # ----------------------------------------------------------------
    # Parse the source ONCE to read config pragmas before resolving
    # any runtime settings.  Pragma values override config; CLI overrides
    # pragma (CLI > pragma > config).
    # ----------------------------------------------------------------
    prepared = WorkflowRuntime.prepare(source)
    pragmas = prepared.config_pragmas

    # Resolve strict_json: CLI > pragma > config.
    strict_json = _first(
        args.strict_json,
        _typed_pragma(pragmas, "strict_json", bool),
        config.strict_json,
    )
    # config.strict_json is always a bool, so _first always returns a bool here.
    assert strict_json is not None
    resolved_strict_json: bool = strict_json

    # Resolve loop limit: CLI > pragma > config.
    loop_limit = _first(
        args.max_iters,
        _typed_pragma(pragmas, "max_iters", int),
        config.default_loop_limit,
    )
    assert loop_limit is not None
    resolved_loop_limit: int = loop_limit

    # Resolve timeout: pragma > config (no CLI flag for timeout).
    pragma_timeout_raw = pragmas.get("timeout")
    if pragma_timeout_raw is not None:
        try:
            resolved_timeout: float | None = parse_timeout(str(pragma_timeout_raw))
        except ValueError as exc:
            print(f"Error: invalid pragma timeout: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    else:
        resolved_timeout = config.timeout

    # Resolve + validate the trace log file up front (F2a/F6).  --dry-run is
    # side-effect-free: no trace is written regardless of --log-file (plan §10.1).
    # Pragma log/log_file values are wired here (Milestone 3).
    if dry_run.enabled():
        log_file = None
    else:
        log_file = prepare_trace_log_from_layers(
            command_name="exec",
            cli_no_log=args.no_log,
            cli_log=args.log,
            cli_log_file=args.log_file,
            pragma_log=_typed_pragma(pragmas, "log", bool),
            pragma_log_file=_typed_pragma(pragmas, "log_file", str),
            config_log=config.log,
            config_log_file=config.log_file,
        )

    # ----------------------------------------------------------------
    # Resolve the runner command: CLI flag > pragma > [exec] config > shared
    # loop default (the same default used by agm loop/review, per plan §9.5).
    # ----------------------------------------------------------------
    runner_cmd = (
        args.runner
        or _typed_pragma(pragmas, "runner", str)
        or config.runner
        or default_agent_runner(merged=merged_config)
    )

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
    # ``prepare`` was already called above to read config pragmas; the same
    # ``PreparedProgram`` is reused here and handed to ``run_prepared`` below,
    # so the program is never parsed or scoped twice.  On a source with
    # parse/scope errors ``declared_agents`` is ``()`` and ``run_prepared``
    # resurfaces the captured diagnostic (exit 1).
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
        idle_timeout=resolved_timeout,
    )

    runtime = WorkflowRuntime(
        default_loop_limit=resolved_loop_limit,
        default_strict_json=resolved_strict_json,
        default_agent=factory,
        shell_exec_timeout=resolved_timeout,
    )

    # Register every declared agent so the registered set equals the declared
    # set: M4 reconciliation always passes; config-only agents the source never
    # declares stay inert (NOT registered), per plan §9.
    for d in decls:
        runtime.register_agent(d.name, factory)

    discovery = runtime.discover_params(prepared)
    for diag in discovery.warnings:
        print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)

    external_params: dict[str, object] = {}
    checked = discovery.checked
    if checked is not None:
        collision_errors = check_param_collisions(
            discovery.params, source_name=diagnostic_source_name
        )
        if collision_errors:
            for err in collision_errors:
                print(f"Error: {err}", file=sys.stderr)
            raise SystemExit(1)
        try:
            cli_params = parse_param_tokens(discovery.params, args.param_tokens)
        except ValueError as exc:
            exit_with_usage_error(["exec"], f"error: {exc}")

        if discovery.program_name is not None:
            param_config_key: str | None = discovery.program_name
        elif args.file is not None:
            param_config_key = Path(args.file).stem
        else:
            param_config_key = None

        config_param_values = (
            params_config_from_merged(merged_config, param_config_key)
            if param_config_key is not None
            else {}
        )
        declared_names = {p.name for p in discovery.params}
        resolved_params, config_warnings = resolve_param_values(
            declared_names,
            config_param_values,
            cli_params,
            program_name=param_config_key,
        )
        external_params.update(resolved_params)
        for msg in config_warnings:
            print(msg, file=sys.stderr)

    # Reuse the ``PreparedProgram`` from above — no second parse/scope of the source.
    result = runtime.run_prepared(
        prepared,
        param_values=external_params,
        check_only=dry_run.enabled(),
        log_file=log_file,
        checked=checked,
    )

    # Warnings live on their own channel and never affect the exit code;
    # ``result.diagnostics`` holds only error-severity pre-execution failures.
    # Warnings carry a ``warning:`` prefix to disambiguate them from error
    # diagnostics on the shared stderr channel (F8).
    printed_warnings = {
        (
            diag.line,
            diag.column,
            diag.end_line,
            diag.end_column,
            diag.message,
            diag.severity,
        )
        for diag in discovery.warnings
    }
    for diag in result.warnings:
        warning_key = (
            diag.line,
            diag.column,
            diag.end_line,
            diag.end_column,
            diag.message,
            diag.severity,
        )
        if warning_key not in printed_warnings:
            print(
                format_diagnostic(diag, source_name=diagnostic_source_name),
                file=sys.stderr,
            )

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
            print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
        raise SystemExit(1)

    # Uncaught AgL exception: print and exit 2 (design §12.6: include source
    # location and trace_id in the error line so the caller can correlate the
    # error with the trace file and the source program).
    print(result.error.to_message(include_trace_id=True), file=sys.stderr)
    raise SystemExit(2)
