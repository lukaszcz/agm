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
    - ``--agent AGL_LITERAL`` seeds ``std/config::default-agent`` from one typed
      constant Agent expression.
    - Every loaded entry and library module opens ``std/core`` by default
      (except ``std/core`` itself). ``--no-stdlib`` disables that automatic
      opening throughout the loaded program. Ordinary imports are qualified by
      default; ``open import`` and ``using`` make selected names bare.
    - A program reads and writes the engine settings (``strict-json``,
      ``max-iters``, ``default-agent``, ``timeout``, ``log``, ``log-file``) through the
      ``std/config`` module; a ``std/config::KEY := VALUE`` write takes effect
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

from agm.agl import PipelineDriver
from agm.agl.diagnostics import format_diagnostic
from agm.agl.modules.roots import assemble_roots
from agm.agl.runtime.agents import value_driven_agent_factory
from agm.agl.runtime.host_settings import HostSettingsPolicy
from agm.agl.semantics.engine_keys import ENGINE_KEY_NAMES, RESERVED_PROGRAM_NAMES
from agm.cli_support.args import ExecArgs
from agm.cli_support.engine_seeds import build_host_engine_seeds, check_max_iters
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
from agm.core.log import (
    LiveTracePathResolver,
    prepare_trace_log_from_decision,
    resolve_log_decision,
)
from agm.core.parse import parse_timeout
from agm.core.toml import toml_dict
from agm.parser import exit_with_usage_error

_T = TypeVar("_T")


def _first(*values: _T | None) -> _T | None:
    """Return the first non-None value, or None if all are None."""
    return next((v for v in values if v is not None), None)


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
    # ``std/config::KEY := VALUE`` write takes effect at its program point and
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
    # resolution. The declared ``program NAME`` takes precedence over the file stem;
    # a stem in the stable reserved-program-name set produces no key (and triggers
    # the reserved-stem error below when no ``program NAME`` declaration exists).
    if prepared.program_name is not None:
        program_key: str | None = prepared.program_name
    elif raw_stem is not None and raw_stem not in RESERVED_PROGRAM_NAMES:
        program_key = raw_stem
    else:
        program_key = None

    # Reserved file-stem check: when no ``program NAME`` declaration is present,
    # a reserved stem would select a conflicting or invalid program config key.
    # Require an explicit non-reserved name instead. Inline ``-c`` (no file stem)
    # is unaffected.
    if (
        raw_stem is not None
        and raw_stem in RESERVED_PROGRAM_NAMES
        and prepared.program_name is None
    ):
        print(
            f"Error: file stem '{raw_stem}' is a reserved program name. "
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

    # Resolve strict_json: CLI > config. A source ``std/config::strict-json :=
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
    # valve off. A source ``std/config::max-iters := VALUE`` write is applied
    # at runtime from its program point, overriding this initial value.
    check_max_iters(args.max_iters)
    resolved_loop_limit = _first(args.max_iters, config.default_loop_limit)

    # Resolve timeout: CLI > [exec] config. A source ``std/config::timeout :=
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

    factory = value_driven_agent_factory(idle_timeout=resolved_timeout)

    # ``prepare_program`` was already called above; the same ``PreparedProgram`` is
    # reused for discovery and the run, so the source is loaded and scoped only once.
    runtime = PipelineDriver(
        default_loop_limit=resolved_loop_limit,
        default_strict_json=resolved_strict_json,
        agent_dispatcher=factory,
        shell_exec_timeout=resolved_timeout,
        default_call_depth_limit=resolved_call_depth_limit,
    )
    discovery = runtime.discover_params(prepared)
    for diag in discovery.warnings:
        print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
    checked = discovery.checked
    if checked is None:
        for diag in discovery.diagnostics:
            print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
        raise SystemExit(1)

    external_params: dict[str, object] = {}
    collision_errors = check_param_collisions(discovery.params, source_name=diagnostic_source_name)
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

    # Params are validated against the lowered program, so this preflight lowers
    # the graph.  It must report a param failure (exit 1) BEFORE the trace file
    # is prepared and the runner is built — hence a check-only pass here rather
    # than letting the run below surface it.  The lowered program it produces is
    # handed to that run, so the graph is lowered exactly once per invocation.
    param_preflight = runtime.preflight_params(
        prepared,
        param_values=external_params,
        compiled=discovery.compiled,
    )
    if not param_preflight.result.ok:
        for diag in param_preflight.result.diagnostics:
            print(format_diagnostic(diag, source_name=diagnostic_source_name), file=sys.stderr)
        raise SystemExit(1)

    # Resolve the CLI > config logging decision ONCE: it both drives the trace
    # file prepared here and seeds the readable ``log`` register below.
    log_decision = resolve_log_decision(
        cli_no_log=args.no_log,
        cli_log=args.log,
        cli_log_file=args.log_file,
        config_log=config.log,
        config_log_file=config.log_file,
    )

    # Resolve + validate the trace log file up front.  --dry-run is
    # side-effect-free: no trace is written regardless of --log-file.  A source
    # ``std/config::log``/``log-file`` write takes effect at runtime via the host
    # reconfigurer, not here.
    if dry_run.enabled():
        log_file = None
    else:
        log_file = prepare_trace_log_from_decision(log_decision, command_name="exec")

    policy = HostSettingsPolicy(
        resolve_trace_path=LiveTracePathResolver(command_name="exec", auto_path=log_file),
    )

    # Seed only settings explicitly controlled by CLI/config. Runtime fallbacks
    # are not seeds: passing them here would suppress a declared
    # ``builtin var`` initializer.  The shared decoder preserves explicit
    # ``None`` values for Option settings such as --no-timeout.
    builtin_host_settings = build_host_engine_seeds(
        config=config,
        primary_table=program_table,
        fallback_table=toml_dict(merged_config.get("exec")),
        log_enabled=log_decision.enabled,
        strict_json=args.strict_json,
        max_iters=args.max_iters,
        log=args.log,
        no_log=args.no_log,
        log_file=args.log_file,
        agent=args.agent,
        timeout=args.timeout,
        no_timeout=args.no_timeout,
        no_log_file=args.no_log_file,
    )

    # Reuse the ``PreparedProgram`` from above — no second parse/scope of the source.
    # Pass the already-computed compiled from discovery and the program the
    # preflight already lowered, so the graph is type-checked, match-compiled and
    # lowered exactly once.
    result = runtime.run_prepared(
        prepared,
        param_values=external_params,
        check_only=dry_run.enabled(),
        log_file=log_file,
        compiled=discovery.compiled,
        executable=param_preflight.executable,
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
