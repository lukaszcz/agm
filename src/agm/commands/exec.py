"""Implementation of the ``agm exec FILE`` command.

Behaviour: read the ``.agl`` source — either from the inline ``-c/--command``
argument or from the source file (exit 1 if unreadable), load the
``[exec]`` configuration, construct a ``PipelineDriver`` with the resolved
settings, call ``runtime.run`` (or a static-only dry run under ``--dry-run``),
print diagnostics to stderr, and exit per the exit-code contract.

Warnings (``result.warnings``) and error diagnostics (``result.diagnostics``)
are two separate channels: warnings are printed to stderr like errors but never
affect the exit code; only error-severity diagnostics yield exit 1.  The
diagnostic severity is included in compiler-style output, e.g.
``path.agl:1:5: warning: message`` or ``1:5: error: message`` for inline
``-c/--command`` source.

Exit-code contract:
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
      of these three flags may be given (mutually exclusive).  ``[exec] log =
      true`` in config also enables logging; CLI flags override config.
    - ``--runner COMMAND`` overrides the default agent runner command from config.
      When set, it is used as the default runner for all unnamed agents.
    - A program reads and writes the engine settings (``strict-json``,
      ``max-iters``, ``runner``, ``timeout``, ``log``, ``log-file``) through the
      ``std.config`` module; a ``std.config::KEY := VALUE`` write takes effect
      from its program point onward and overrides the CLI flag, which overrides
      the config-file layer.  ``--max-call-depth`` remains a host/runtime
      recursion guard.
    - ``--dry-run`` (global flag) runs only the static pipeline + contract
      materialization and never writes a trace.  Evaluation and extern
      companion imports are skipped, so broken companion Python files do not
      fail a dry run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypeVar

from agm.agent.config import default_agent_runner
from agm.agent.runner import parse_command, split_command
from agm.agl import PipelineDriver
from agm.agl.diagnostics import format_diagnostic
from agm.agl.modules.roots import assemble_roots
from agm.agl.runtime.agents import AgentFn, runner_backed_agent_factory
from agm.agl.runtime.host_settings import HostSettingsPolicy
from agm.agl.runtime.params import convert_config_value, raw_option_str
from agm.agl.semantics.engine_keys import (
    ENGINE_KEY_NAMES,
    RESERVED_PROGRAM_NAMES,
    get_engine_key_type,
)
from agm.agl.semantics.types import Type
from agm.agl.semantics.values import BoolValue, TextValue, Value
from agm.cli_support.args import ExecArgs
from agm.cli_support.exec_params import (
    check_param_collisions,
    parse_param_tokens,
    resolve_param_values,
)
from agm.config.context import current_config_context
from agm.config.general import (
    exec_config_from_merged,
    load_merged_config,
    program_config_from_merged,
)
from agm.config.module_roots import load_module_roots, resolve_lib_root, resolve_stdlib_root
from agm.core import dry_run
from agm.core.fs import mkdir, read_text_arg
from agm.core.log import prepare_trace_log_from_layers, resolve_log_file
from agm.core.parse import format_timeout, parse_timeout
from agm.core.toml import toml_dict
from agm.parser import exit_with_usage_error

_T = TypeVar("_T")


def _first(*values: _T | None) -> _T | None:
    """Return the first non-None value, or None if all are None."""
    return next((v for v in values if v is not None), None)


def _project_option_text(
    value: str | None, *, no_flag: bool, key_name: str, key_type: Type
) -> Value | None:
    """Project an Option[text] CLI flag pair to an EnumValue or None.

    - *value* is set: ``some(value)`` via :func:`convert_config_value`.
    - *no_flag* is ``True``: ``none`` via :func:`convert_config_value`.
    - Both absent: returns ``None``.
    """
    if value is not None:
        return convert_config_value(key_name, value, key_type)
    if no_flag:
        return convert_config_value(key_name, None, key_type)
    return None


def run(args: ExecArgs) -> None:
    """Run the ``agm exec`` command."""
    # The program source comes either from an inline ``-c/--command`` argument
    # or from a file.  The CLI layer guarantees exactly one is provided; the
    # defensive ``else`` keeps ``run`` safe when called directly.
    if args.command is not None:
        source = args.command
        entry_path: Path | None = None
        diagnostic_source_name: str | None = None
    elif args.file is not None:
        source = read_text_arg(Path(args.file))
        entry_path = Path(args.file)
        diagnostic_source_name = args.file
    else:
        print("Error: exec requires either a FILE or -c/--command", file=sys.stderr)
        raise SystemExit(1)

    # Load the merged config once; program_table and exec config are deferred until
    # after prepare_program so the declared program name is available.
    ctx = current_config_context()
    raw_stem: str | None = Path(args.file).stem if args.file is not None else None
    merged_config = load_merged_config(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)

    # ----------------------------------------------------------------
    # Assemble module roots and load + scope the graph ONCE.  A source
    # ``std.config::KEY := VALUE`` write takes effect at its program point and
    # overrides the CLI flag, which overrides the config-file layer.
    # ----------------------------------------------------------------
    try:
        mr_config = load_module_roots(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)
    except ValueError as exc:
        print(f"Error: invalid module roots configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # Resolve the lib_root path.
    stdlib_root = resolve_stdlib_root(home=ctx.home)
    resolved_lib_root = resolve_lib_root(mr_config, home=ctx.home)

    # The invocation root is the entry file's directory (for file exec) or
    # the cwd (for -c inline exec).
    if entry_path is not None:
        invocation_root = entry_path.parent
    else:
        invocation_root = ctx.cwd

    roots = assemble_roots(
        invocation_root=invocation_root,
        stdlib_root=stdlib_root,
        lib_root=resolved_lib_root,
        configured=mr_config.extra,
        cli=args.module_paths,
        cwd=ctx.cwd,
    )

    prepared = PipelineDriver.prepare_program(
        source, entry_path=entry_path, roots=roots, default_stdlib=not args.no_stdlib
    )

    # Resolve the single final program key for BOTH engine-key overrides and param
    # resolution.  The declared ``program NAME`` takes precedence over the file stem;
    # a stem that collides with an AGM reserved section name produces no key (and
    # triggers the reserved-stem error below when no ``program NAME`` decl exists).
    if prepared.program_name is not None:
        program_key: str | None = prepared.program_name
    elif raw_stem is not None and raw_stem not in RESERVED_PROGRAM_NAMES:
        program_key = raw_stem
    else:
        program_key = None

    # Reserved file-stem check: when no ``program NAME`` decl is present,
    # a file stem that matches an AGM config section name would silently shadow
    # the global ``[exec]`` config.  Require an explicit ``program NAME`` decl
    # with a non-reserved name instead.  Inline ``-c`` (no file stem) is unaffected.
    if (
        raw_stem is not None
        and raw_stem in RESERVED_PROGRAM_NAMES
        and prepared.program_name is None
    ):
        print(
            f"Error: file stem '{raw_stem}' is a reserved AGM section name. "
            "Add a 'program NAME' declaration with a non-reserved name.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Fetch the [<program_key>] table once.  The same table feeds both the
    # engine-key overlay in exec_config_from_merged and param resolution in
    # resolve_param_values, so both always read the same program section.
    program_table: dict[str, object] = (
        program_config_from_merged(merged_config, program_key) if program_key is not None else {}
    )
    try:
        config = exec_config_from_merged(merged_config, program_table=program_table)
    except ValueError as exc:
        print(f"Error: invalid exec configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # The log-file engine-key type is needed to seed the host-consumed
    # ``log-file`` register from the resolved CLI/config layers.
    _log_file_type = get_engine_key_type("log-file")
    assert _log_file_type is not None  # always a valid engine key

    base_runner_cmd = config.runner or default_agent_runner(merged=merged_config)

    # Resolve strict_json: CLI > config. A source ``std.config::strict-json :=
    # VALUE`` write is applied at runtime when it updates the live setting.
    strict_json = _first(args.strict_json, config.strict_json)
    # config.strict_json is always a bool, so _first always returns a bool here.
    assert strict_json is not None
    resolved_strict_json: bool = strict_json

    # Resolve max call depth: CLI > config.  ``None`` (nothing set at
    # any layer) lets the driver apply its canonical default.
    resolved_call_depth_limit = _first(
        args.max_call_depth,
        config.max_call_depth,
    )

    # Resolve loop limit (max-iters valve): CLI > config. ``None`` leaves the
    # valve off. A source ``std.config::max-iters := VALUE`` write is applied
    # at runtime from its program point, overriding this initial value.
    if args.max_iters is not None and args.max_iters <= 0:
        print("Error: --max-iters must be a positive integer", file=sys.stderr)
        raise SystemExit(1)
    resolved_loop_limit = _first(args.max_iters, config.default_loop_limit)

    # Resolve timeout: CLI > [exec] config. A source ``std.config::timeout :=
    # VALUE`` write is applied at runtime from its program point.
    # ``--timeout VALUE`` overrides the config; ``--no-timeout`` clears it (None).
    if args.timeout is not None:
        try:
            resolved_timeout: float | None = parse_timeout(args.timeout)
        except ValueError as exc:
            print(f"Error: invalid --timeout value: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    elif args.no_timeout:
        resolved_timeout = None
    else:
        resolved_timeout = config.timeout

    # ----------------------------------------------------------------
    # Resolve declared agents and wire each one against a single runner-backed
    # factory whose per-agent command map merges in precedence order (high →
    # low):
    #
    #     [exec.agents.<name>]   (config, per-agent)
    #     source `agent` runner hint
    #     resolved default runner (runner_cmd, the floor)
    #
    # The source program OWNS the agent name set: every named agent must be
    # declared.  One factory backs ``prompt`` (the default) and every declared
    # name; it dispatches by ``request.agent`` against ``per_agent_cmds``,
    # falling back to the default runner (the floor).  The agent idle-timeout is
    # start-resolved from CLI > [exec] config > engine default and fixed for the
    # lifetime of this factory.  A source ``std.config::timeout := e`` write
    # updates ONLY the live shell-exec timeout from its program point onward.
    # The runner command resolves CLI flag > [exec] config > shared loop default
    # (the same default used by agm loop/review).
    decls = prepared.declared_agents
    source_hints = {d.name: d.runner for d in decls if d.runner is not None}
    per_agent_cmds = {**source_hints, **config.agents}
    runner_cmd = args.runner or config.runner or base_runner_cmd
    # Validate the resolved runner command eagerly: malformed quoting (e.g.
    # unclosed quote) and whitespace-only values are caught here via
    # split_command, which handles the ValueError from shlex.split.  This
    # honours the exit-1 = pre-execution contract.
    split_command(runner_cmd, kind="runner")
    for declaration in decls:
        cmd = per_agent_cmds.get(declaration.name)
        if cmd is not None:
            split_command(cmd, kind="runner")

    factory = runner_backed_agent_factory(
        default_runner_cmd=runner_cmd,
        per_agent_cmds=per_agent_cmds,
        idle_timeout=resolved_timeout,
    )

    # ``prepare_program`` was already called above; the same ``PreparedGraph`` is
    # reused for discovery and the run, so the source is loaded and scoped only
    # once.  On a source with load/scope errors ``declared_agents`` is ``()`` and
    # ``run_prepared_graph`` resurfaces the captured diagnostic (exit 1).
    runtime = PipelineDriver(
        default_loop_limit=resolved_loop_limit,
        default_strict_json=resolved_strict_json,
        default_agent=factory,
        shell_exec_timeout=resolved_timeout,
        default_call_depth_limit=resolved_call_depth_limit,
    )
    # Register every declared agent so the registered set equals the declared
    # set: reconciliation always passes; config-only agents the source never
    # declares stay inert (NOT registered).
    for declaration in decls:
        if declaration.name in per_agent_cmds:
            runtime.register_agent(declaration.name, factory)

    discovery = runtime.discover_params_graph(prepared)
    for diag in discovery.warnings:
        print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
    checked = discovery.checked
    if checked is None:
        for diag in discovery.diagnostics:
            print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
        raise SystemExit(1)

    external_params: dict[str, object] = {}
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

    config_param_values = {k: v for k, v in program_table.items() if k not in ENGINE_KEY_NAMES}
    declared_names = {p.name for p in discovery.params}
    resolved_params, config_warnings = resolve_param_values(
        declared_names,
        config_param_values,
        cli_params,
        program_name=program_key,
    )
    external_params.update(resolved_params)
    for msg in config_warnings:
        print(msg, file=sys.stderr)

    param_preflight = runtime.run_prepared_graph(
        prepared,
        param_values=external_params,
        check_only=True,
        compiled_graph=discovery.compiled_graph,
    )
    if not param_preflight.ok:
        for diag in param_preflight.diagnostics:
            print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
        raise SystemExit(1)

    # Resolve + validate the trace log file up front.  --dry-run is
    # side-effect-free: no trace is written regardless of --log-file.  A source
    # ``std.config::log``/``log-file`` write takes effect at runtime via the host
    # reconfigurer, not here.
    if dry_run.enabled():
        log_file = None
    else:
        log_file = prepare_trace_log_from_layers(
            command_name="exec",
            cli_no_log=args.no_log,
            cli_log=args.log,
            cli_log_file=args.log_file,
            config_log=config.log,
            config_log_file=config.log_file,
        )

    # Host policy for reflecting host-consumed ``builtin var`` writes
    # (``runner``, ``log``, ``log-file``) into the live services during the run.
    # A source ``std.config::runner := ...`` rebuilds the default agent from the
    # new command (source-authoritative); ``log``/``log-file`` writes repoint the
    # trace store.  The mid-run trace repoint must NOT truncate an existing file,
    # so it resolves the path directly rather than through ``prepare_trace_log``.
    def _build_runner(command: str) -> AgentFn:
        parse_command(command, kind="runner")
        return runner_backed_agent_factory(
            default_runner_cmd=command,
            per_agent_cmds=per_agent_cmds,
            idle_timeout=resolved_timeout,
        )

    def _resolve_trace_path(enabled: bool, log_file_value: str | None) -> Path | None:
        path = resolve_log_file(
            command_name="exec",
            enabled=enabled,
            log_file=log_file_value,
            unique=True,
        )
        if path is not None:
            mkdir(path.parent, parents=True, exist_ok=True)
        return path

    policy = HostSettingsPolicy(
        build_runner=_build_runner, resolve_trace_path=_resolve_trace_path
    )

    # Seed the host-consumed registers from the resolved host layers so a program
    # reading these settings before any write observes the effective start value.
    # A later source ``:=`` overrides the seed from its program point onward.
    if args.no_log:
        seed_log = False
    elif args.log or args.log_file is not None:
        seed_log = True
    else:
        seed_log = config.log or config.log_file is not None
    seed_log_file = _project_option_text(
        args.log_file, no_flag=args.no_log_file, key_name="log-file", key_type=_log_file_type
    )
    if seed_log_file is None:
        seed_log_file = convert_config_value("log-file", config.log_file, _log_file_type)
    _timeout_type = get_engine_key_type("timeout")
    assert _timeout_type is not None
    exec_raw_table = toml_dict(merged_config.get("exec"))
    if args.timeout is not None:
        timeout_seed_raw: object = args.timeout
    elif args.no_timeout:
        timeout_seed_raw = None
    else:
        timeout_seed_raw = raw_option_str(program_table, exec_raw_table, "timeout")
        if timeout_seed_raw is None and resolved_timeout is not None:
            timeout_seed_raw = format_timeout(resolved_timeout)
    builtin_host_settings: dict[str, Value] = {
        "runner": TextValue(runner_cmd),
        "log": BoolValue(seed_log),
        "log-file": seed_log_file,
        "timeout": convert_config_value("timeout", timeout_seed_raw, _timeout_type),
    }

    # Reuse the ``PreparedGraph`` from above — no second parse/scope of the source.
    # Pass the already-computed compiled_graph from discovery so the graph is
    # type-checked and match-compiled exactly once.
    result = runtime.run_prepared_graph(
        prepared,
        param_values=external_params,
        check_only=dry_run.enabled(),
        log_file=log_file,
        compiled_graph=discovery.compiled_graph,
        host_settings_policy=policy,
        builtin_host_settings=builtin_host_settings,
    )

    # Warnings live on their own channel and never affect the exit code;
    # ``result.diagnostics`` holds only error-severity pre-execution failures.
    # Warnings carry a ``warning:`` prefix to disambiguate them from error
    # diagnostics on the shared stderr channel.
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

    # Uncaught AgL exception: print and exit 2.
    print(result.error.to_message(include_trace_id=True), file=sys.stderr)
    raise SystemExit(2)
