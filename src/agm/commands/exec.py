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
      of these three flags may be given (mutually exclusive).  A source
      ``config log = true`` declaration or ``[exec] log = true`` in
      config also enables logging; CLI flags override source declarations
      (CLI > source > config).
    - ``--runner COMMAND`` overrides the default agent runner command from config.
      When set, it is used as the default runner for all unnamed agents.
    - Source ``config KEY = VALUE`` declarations override config-file settings for
      ``strict-json``, ``max-iters``, ``runner``, ``timeout``, ``log``, and
      ``log-file``.  CLI flags always take precedence.
      ``--max-call-depth`` remains a host/runtime recursion guard.
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
from agm.agent.runner import split_command
from agm.agl import PipelineDriver
from agm.agl.diagnostics import format_diagnostic
from agm.agl.modules.roots import assemble_roots
from agm.agl.runtime.agents import runner_backed_agent_factory
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.runtime.params import (
    build_engine_config_base,
    convert_config_value,
    raw_option_str,
)
from agm.agl.semantics.engine_keys import (
    ENGINE_KEY_NAMES,
    RESERVED_PROGRAM_NAMES,
    get_engine_key_type,
)
from agm.agl.semantics.types import Type
from agm.agl.semantics.values import BoolValue, EnumValue, IntValue, TextValue, Value
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
from agm.core.fs import read_text_arg
from agm.core.log import prepare_trace_log_from_layers
from agm.core.parse import parse_timeout
from agm.core.toml import toml_dict
from agm.parser import exit_with_usage_error

_T = TypeVar("_T")


def _first(*values: _T | None) -> _T | None:
    """Return the first non-None value, or None if all are None."""
    return next((v for v in values if v is not None), None)


def _bool_value(values: dict[str, Value], key: str) -> bool | None:
    """Return a bool source config value payload for *key*."""
    value = values.get(key)
    if isinstance(value, BoolValue):
        return value.value
    return None


def _text_value(values: dict[str, Value], key: str) -> str | None:
    """Return a text source config value payload for *key*."""
    value = values.get(key)
    if isinstance(value, TextValue):
        return value.value
    return None


def _option_text_value(values: dict[str, Value], key: str) -> str | None:
    """Return the ``some(text)`` payload for an Option[text] source config value."""
    value = values.get(key)
    if isinstance(value, EnumValue) and value.variant == "Some":
        raw = value.fields.get("value")
        if isinstance(raw, TextValue):
            return raw.value
    return None


# ---------------------------------------------------------------------------
# Centralized config_cli projection helpers
# ---------------------------------------------------------------------------
# Each helper returns a ``Value`` when the CLI flag was supplied, or ``None``
# when it was absent (the key should not appear in ``config_cli``).


def _project_bool_pair(positive: bool, negative: bool) -> Value | None:
    """Project a pair of exclusive bool flags (--X / --no-X) to a BoolValue or None.

    Returns ``BoolValue(True)`` when *positive* is set, ``BoolValue(False)`` when
    *negative* is set, and ``None`` when neither is set (absent from ``config_cli``).
    """
    if positive:
        return BoolValue(True)
    if negative:
        return BoolValue(False)
    return None


def _project_option_text(
    value: str | None, *, no_flag: bool, key_name: str, key_type: Type
) -> Value | None:
    """Project an Option[text] CLI flag pair to an EnumValue or None.

    - *value* is set: ``some(value)`` via :func:`convert_config_value`.
    - *no_flag* is ``True``: ``none`` via :func:`convert_config_value`.
    - Both absent: returns ``None`` (key absent from ``config_cli``).
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
    # Assemble module roots and load + scope the graph ONCE to read source
    # config declarations before resolving any runtime settings.  Source values
    # override config; CLI overrides source (CLI > source > config).
    # ----------------------------------------------------------------
    try:
        mr_config = load_module_roots(
            home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd
        )
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

    # Build the config-binding resolution maps consumed by ``IrConfigBind``.
    # They are needed both for the startup-config prepass and for the normal run.
    _timeout_type = get_engine_key_type("timeout")
    assert _timeout_type is not None  # always a valid engine key
    _log_file_type = get_engine_key_type("log-file")
    assert _log_file_type is not None  # always a valid engine key

    config_cli: dict[str, Value] = {}
    if args.strict_json is not None:
        config_cli["strict-json"] = BoolValue(args.strict_json)
    if args.max_iters is not None:
        config_cli["max-iters"] = IntValue(args.max_iters)
    if args.runner is not None:
        config_cli["runner"] = TextValue(args.runner)
    if (v := _project_bool_pair(args.log, args.no_log)) is not None:
        config_cli["log"] = v
    if (
        v := _project_option_text(
            args.log_file,
            no_flag=args.no_log_file,
            key_name="log-file",
            key_type=_log_file_type,
        )
    ) is not None:
        config_cli["log-file"] = v
    if (
        v := _project_option_text(
            args.timeout,
            no_flag=args.no_timeout,
            key_name="timeout",
            key_type=_timeout_type,
        )
    ) is not None:
        config_cli["timeout"] = v

    base_runner_cmd = config.runner or default_agent_runner(merged=merged_config)
    exec_raw_table = toml_dict(merged_config.get("exec"))
    raw_timeout = raw_option_str(program_table, exec_raw_table, "timeout")
    # The config_base floor for ``max-iters`` is the resolved config value when
    # set, else the engine default (5) — a bare ``config max-iters`` decl binds
    # this floor and turns the valve ON.  Omit the key when unset so
    # build_engine_config_base falls back to its _ENGINE_DEFAULTS entry.
    engine_base_raw: dict[str, object] = {
        "strict-json": config.strict_json,
        "runner": base_runner_cmd,
        "log": config.log,
        "timeout": raw_timeout,
        "log-file": config.log_file,
    }
    if config.default_loop_limit is not None:
        engine_base_raw["max-iters"] = config.default_loop_limit
    config_base = build_engine_config_base(engine_base_raw)

    # Resolve strict_json: CLI > config. Source config strict-json = VALUE is
    # applied at runtime when IrConfigBind updates the live interpreter setting.
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

    # Resolve loop limit (max-iters valve): CLI > config. ``None`` (nothing set
    # at any layer) means the valve is OFF — unguarded loops run until they
    # self-terminate. Source ``config max-iters = VALUE`` is applied at runtime
    # when the config binding executes, overriding this initial value.
    resolved_loop_limit: int | None = _first(args.max_iters, config.default_loop_limit)

    # Resolve timeout: CLI > [exec] config. Source config timeout = VALUE is
    # applied at runtime when the config binding executes.
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

    # Startup config runs before a source ``config runner`` value can choose
    # the final default runner. It therefore dispatches agents through the
    # resolved CLI/config/default floor. Named agents retain their normal
    # config > source-hint > default precedence during that bootstrap pass.
    decls = prepared.declared_agents
    source_hints = {d.name: d.runner for d in decls if d.runner is not None}
    per_agent_cmds = {**source_hints, **config.agents}
    bootstrap_runner_cmd = args.runner or config.runner or base_runner_cmd
    split_command(bootstrap_runner_cmd, kind="runner")
    for declaration in decls:
        cmd = per_agent_cmds.get(declaration.name)
        if cmd is not None:
            split_command(cmd, kind="runner")
    bootstrap_factory = runner_backed_agent_factory(
        default_runner_cmd=bootstrap_runner_cmd,
        per_agent_cmds=per_agent_cmds,
        idle_timeout=resolved_timeout,
    )

    # Parameter discovery and validation are static.  They must complete before
    # evaluating startup config because that config may call an agent to compute
    # its runner, log, or log-file value.
    discovery_runtime = PipelineDriver(
        default_loop_limit=resolved_loop_limit,
        default_strict_json=resolved_strict_json,
        default_agent=bootstrap_factory,
        shell_exec_timeout=resolved_timeout,
        default_call_depth_limit=resolved_call_depth_limit,
    )
    for declaration in decls:
        discovery_runtime.register_agent(declaration.name, bootstrap_factory)
    discovery = discovery_runtime.discover_params_graph(prepared)
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

        config_param_values = {
            k: v for k, v in program_table.items() if k not in ENGINE_KEY_NAMES
        }
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

        param_preflight = discovery_runtime.run_prepared_graph(
            prepared,
            param_values=external_params,
            check_only=True,
            compiled_graph=discovery.compiled_graph,
            config_cli=config_cli,
            config_base=config_base,
        )
        if not param_preflight.ok:
            for diag in param_preflight.diagnostics:
                print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
            raise SystemExit(1)

    shared_extern_registry = ExternRegistry()
    # The startup pass must advertise the same declared-agent capabilities as
    # the execution pass so its compiled artifact remains reusable.
    startup_values: dict[str, Value] = {}
    if prepared.resolved_graph is not None and not dry_run.enabled():
        startup_runtime = PipelineDriver(
            default_loop_limit=resolved_loop_limit,
            default_strict_json=resolved_strict_json,
            default_agent=bootstrap_factory,
            shell_exec_timeout=resolved_timeout,
        )
        startup_runtime.share_extern_registry(shared_extern_registry)
        for declaration in decls:
            startup_runtime.register_agent(declaration.name, bootstrap_factory)
        startup_result = startup_runtime.collect_startup_config_graph(
            prepared,
            names={"runner", "log", "log-file"},
            param_values=external_params,
            config_cli=config_cli,
            config_base=config_base,
        )
        if startup_result.diagnostics:
            for diag in startup_result.warnings:
                print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
            for diag in startup_result.diagnostics:
                print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
            raise SystemExit(1)
        if startup_result.error is not None:
            for diag in startup_result.warnings:
                print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
            print(startup_result.error.to_message(include_trace_id=True), file=sys.stderr)
            raise SystemExit(2)
        startup_values = startup_result.values if startup_result.ok else {}

    # Resolve + validate the trace log file up front.  --dry-run is
    # side-effect-free: no trace is written regardless of --log-file.
    # Source config log/log-file values are wired here.
    if dry_run.enabled():
        log_file = None
    else:
        log_file = prepare_trace_log_from_layers(
            command_name="exec",
            cli_no_log=args.no_log,
            cli_log=args.log,
            cli_log_file=args.log_file,
            source_log=_bool_value(startup_values, "log"),
            source_log_file=_option_text_value(startup_values, "log-file"),
            config_log=config.log,
            config_log_file=config.log_file,
        )

    # ----------------------------------------------------------------
    # Resolve the runner command: CLI flag > source constant > [exec] config >
    # shared loop default (the same default used by agm loop/review).
    # ----------------------------------------------------------------
    runner_cmd = (
        args.runner
        or _text_value(startup_values, "runner")
        or config.runner
        or base_runner_cmd
    )

    # Validate the resolved runner command eagerly: malformed quoting (e.g.
    # unclosed quote) and whitespace-only values are caught here via
    # split_command, which also handles the ValueError from shlex.split for
    # malformed quoting.  This honours the exit-1 = pre-execution contract
    # and deduplicates the logic already in split_command.
    split_command(runner_cmd, kind="runner")

    # ----------------------------------------------------------------
    # Resolve declared agents and wire each one explicitly.
    #
    # The source program OWNS the agent name set: every named agent must be
    # declared.  We register each DECLARED agent against a single runner-backed
    # factory whose per-agent command map merges in precedence order
    # (high → low):
    #
    #     [exec.agents.<name>]   (config, per-agent)
    #     source `agent` runner hint
    #     resolved default runner (runner_cmd, the floor)
    #
    # ``prepare_program`` was already called above to read source config declarations; the
    # same ``PreparedGraph`` is reused here and handed to ``run_prepared_graph``
    # below, so the source is never loaded or scoped twice.  On a source with
    # load/scope errors ``declared_agents`` is ``()`` and ``run_prepared_graph``
    # resurfaces the captured diagnostic (exit 1).
    # One factory backs ``prompt`` (the default) and every declared name; it
    # dispatches by ``request.agent`` against ``per_agent_cmds``, falling back
    # to the default runner (the floor).  ``command_with_prompt_target``
    # substitutes ``%%`` / ``%{PROMPT_FILE}`` for source hints and config
    # commands alike.
    # The agent idle-timeout is start-resolved from CLI > [exec] config > engine
    # default and fixed for the lifetime of this factory.  A source
    # ``config timeout = e`` declaration updates ONLY the live shell-exec timeout
    # from its declaration point onward; it does NOT retroactively reconfigure
    # the already-constructed agent factory, because the factory is established
    # before evaluation begins and cannot be reopened mid-run.
    factory = runner_backed_agent_factory(
        default_runner_cmd=runner_cmd,
        per_agent_cmds=per_agent_cmds,
        idle_timeout=resolved_timeout,
    )

    runtime = PipelineDriver(
        default_strict_json=resolved_strict_json,
        default_loop_limit=resolved_loop_limit,
        default_agent=factory,
        shell_exec_timeout=resolved_timeout,
        default_call_depth_limit=resolved_call_depth_limit,
    )
    runtime.share_extern_registry(shared_extern_registry)

    # Register every declared agent so the registered set equals the declared
    # set: reconciliation always passes; config-only agents the source never
    # declares stay inert (NOT registered).
    for d in decls:
        runtime.register_agent(d.name, factory)

    # Reuse the ``PreparedGraph`` from above — no second parse/scope of the source.
    # Pass the already-computed checked_graph from discovery so the graph is
    # type-checked exactly once (mirroring the single-file run_prepared path).
    result = runtime.run_prepared_graph(
        prepared,
        param_values=external_params,
        check_only=dry_run.enabled(),
        log_file=log_file,
        compiled_graph=discovery.compiled_graph,
        config_cli=config_cli,
        config_base=config_base,
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
