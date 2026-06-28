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
    - Source ``config KEY = VALUE`` declarations override config-file settings for
      ``strict-json``, ``max-iters``, ``runner``, ``timeout``, ``log``, and
      ``log-file``.  CLI flags always take precedence.
    - ``--dry-run`` (global flag) runs only the static pipeline + contract
      materialization and never writes a trace (side-effect-free).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from agm.agent.config import default_agent_runner
from agm.agent.runner import split_command
from agm.agl import PipelineDriver
from agm.agl.diagnostics import format_diagnostic
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.modules.roots import assemble_roots
from agm.agl.pipeline import static_config_values
from agm.agl.runtime.agents import runner_backed_agent_factory
from agm.agl.runtime.params import _raw_option_str, convert_config_value
from agm.agl.semantics.engine_keys import (
    ENGINE_KEY_NAMES,
    RESERVED_PROGRAM_NAMES,
    get_engine_key_type,
)
from agm.agl.semantics.types import Type
from agm.agl.semantics.values import BoolValue, IntValue, TextValue, Value
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
_S = TypeVar("_S")
_V = TypeVar("_V", bound=Value)


def _first(*values: _T | None) -> _T | None:
    """Return the first non-None value, or None if all are None."""
    return next((v for v in values if v is not None), None)



def _typed_const(consts: dict[str, bool | int | str], key: str, typ: type[_T]) -> _T | None:
    """Return the static config constant for *key* if present and of type *typ*."""
    value = consts.get(key)
    return value if isinstance(value, typ) else None


# ---------------------------------------------------------------------------
# Centralized config_cli projection helpers
# ---------------------------------------------------------------------------
# Each helper returns a ``Value`` when the CLI flag was supplied, or ``None``
# when it was absent (the key should not appear in ``config_cli``).


def _project_scalar(x: _S | None, ctor: Callable[[_S], _V]) -> _V | None:
    """Project an optional scalar flag to a Value or None using *ctor* to wrap it."""
    return None if x is None else ctor(x)


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
    # Assemble module roots and load + scope the graph ONCE to read config
    # pragmas before resolving any runtime settings.  Pragma values override
    # config; CLI overrides pragma (CLI > pragma > config).
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
    resolved_lib_root = resolve_lib_root(mr_config)

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

    # Source ``config KEY = LITERAL`` declarations contribute compile-time
    # constants for start-resolved keys (``runner``, ``log``, ``log-file``).
    # The three D6 keys (``strict-json``, ``max-iters``, ``timeout``) are NOT
    # folded here; they take effect at the point of the config binding at runtime.
    if prepared.resolved_graph is not None and (
        (entry_mod := prepared.resolved_graph.modules.get(ENTRY_ID)) is not None
    ):
        static_consts = static_config_values(entry_mod.resolved.program)
    else:
        static_consts = {}

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

    # Reserved file-stem check (§15): when no ``program NAME`` decl is present,
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

    # Resolve strict_json: CLI > config.  Source config strict-json = VALUE
    # is applied at runtime by the D6 effect (IrConfigBind → _apply_config_effect).
    strict_json = _first(args.strict_json, config.strict_json)
    # config.strict_json is always a bool, so _first always returns a bool here.
    assert strict_json is not None
    resolved_strict_json: bool = strict_json

    # Resolve loop limit: CLI > config.  Source config max-iters = VALUE is
    # applied at runtime by the D6 effect.
    loop_limit = _first(args.max_iters, config.default_loop_limit)
    assert loop_limit is not None
    resolved_loop_limit: int = loop_limit

    # Resolve timeout: CLI > [exec] config.  Source config timeout = VALUE is
    # applied at runtime by the D6 effect (effect-at-binding).
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

    # Resolve + validate the trace log file up front (F2a/F6).  --dry-run is
    # side-effect-free: no trace is written regardless of --log-file (plan §10.1).
    # Source config log/log-file values are wired here.
    if dry_run.enabled():
        log_file = None
    else:
        log_file = prepare_trace_log_from_layers(
            command_name="exec",
            cli_no_log=args.no_log,
            cli_log=args.log,
            cli_log_file=args.log_file,
            pragma_log=_typed_const(static_consts, "log", bool),
            pragma_log_file=_typed_const(static_consts, "log-file", str),
            config_log=config.log,
            config_log_file=config.log_file,
        )

    # ----------------------------------------------------------------
    # Resolve the runner command: CLI flag > source constant > [exec] config >
    # shared loop default (the same default used by agm loop/review, per §9.5).
    # ----------------------------------------------------------------
    runner_cmd = (
        args.runner
        or _typed_const(static_consts, "runner", str)
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
    # ``prepare_program`` was already called above to read config pragmas; the
    # same ``PreparedGraph`` is reused here and handed to ``run_prepared_graph``
    # below, so the source is never loaded or scoped twice.  On a source with
    # load/scope errors ``declared_agents`` is ``()`` and ``run_prepared_graph``
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
    # The agent idle-timeout is start-resolved from CLI > [exec] config > engine
    # default and fixed for the lifetime of this factory.  A source
    # ``config timeout = e`` declaration updates ONLY the live shell-exec timeout
    # (via effect-at-binding, D6) from its declaration point onward; it does NOT
    # retroactively reconfigure the already-constructed agent factory, because the
    # factory is established before evaluation begins and cannot be reopened
    # mid-run.  This is the §15-sanctioned compromise for the ``timeout`` key.
    factory = runner_backed_agent_factory(
        default_runner_cmd=runner_cmd,
        per_agent_cmds=per_agent_cmds,
        idle_timeout=resolved_timeout,
    )

    runtime = PipelineDriver(
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

    discovery = runtime.discover_params_graph(prepared)
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

        # Strip engine keys before param resolution: they were already applied to
        # ExecConfig; leaving them in would trigger spurious "unknown param" warnings.
        # program_key and program_table are shared with the engine-config path above,
        # ensuring both always read the same [<program>] section.
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

    # Build the config-binding resolution maps consumed by ``IrConfigBind``:
    # ``config_cli`` carries explicitly-set CLI overrides (highest precedence),
    # ``config_base`` carries the host's configured defaults (the fallback for a
    # bare ``config KEY`` with no source value).  Option-typed keys
    # (``timeout``/``log-file``) bind ``some(value)`` when the [exec] config
    # supplies a value, else ``none`` (the engine default).
    #
    # Projection helpers (_project_*) return None when the CLI flag was absent,
    # meaning the key stays out of config_cli and falls through to source/base.
    _timeout_type = get_engine_key_type("timeout")
    assert _timeout_type is not None  # always a valid engine key
    _log_file_type = get_engine_key_type("log-file")
    assert _log_file_type is not None  # always a valid engine key

    config_cli: dict[str, Value] = {}
    # strict-json: tri-state bool (None = absent)
    if (bv := _project_scalar(args.strict_json, BoolValue)) is not None:
        config_cli["strict-json"] = bv
    # max-iters: optional int (None = absent)
    if (iv := _project_scalar(args.max_iters, IntValue)) is not None:
        config_cli["max-iters"] = iv
    # runner: optional text (None = absent)
    if (tv := _project_scalar(args.runner, TextValue)) is not None:
        config_cli["runner"] = tv
    # log: --log/--no-log pair (both False = absent)
    if (v := _project_bool_pair(args.log, args.no_log)) is not None:
        config_cli["log"] = v
    # log-file: Option[text] (None + not no_log_file = absent)
    if (
        v := _project_option_text(
            args.log_file,
            no_flag=args.no_log_file,
            key_name="log-file",
            key_type=_log_file_type,
        )
    ) is not None:
        config_cli["log-file"] = v
    # timeout: Option[text] — use the RAW CLI string, not the pre-parsed float,
    # so the binding holds "30s" rather than "30.0".  If no CLI flag was given,
    # fall through to config_base (which uses str(config.timeout), see below).
    if (
        v := _project_option_text(
            args.timeout,
            no_flag=args.no_timeout,
            key_name="timeout",
            key_type=_timeout_type,
        )
    ) is not None:
        config_cli["timeout"] = v

    # Build config_base with the raw TOML string for Option-typed keys so the
    # binding reflects exactly what was written (e.g. "30s" not "1800.0").
    # For ``timeout``: prefer [<program>].timeout raw string over [exec].timeout.
    # For ``log-file``: config.log_file is already path-resolved from the exec table.
    exec_raw_table = toml_dict(merged_config.get("exec"))
    raw_timeout = _raw_option_str(program_table, exec_raw_table, "timeout")
    config_base: dict[str, Value] = {
        "strict-json": BoolValue(config.strict_json),
        "max-iters": IntValue(config.default_loop_limit),
        "runner": TextValue(runner_cmd),
        "log": BoolValue(config.log),
        "timeout": convert_config_value("timeout", raw_timeout, _timeout_type),
        "log-file": convert_config_value(
            "log-file",
            config.log_file,
            _log_file_type,
        ),
    }

    # Reuse the ``PreparedGraph`` from above — no second parse/scope of the source.
    # Pass the already-computed checked_graph from discovery so the graph is
    # type-checked exactly once (mirroring the single-file run_prepared path).
    result = runtime.run_prepared_graph(
        prepared,
        param_values=external_params,
        check_only=dry_run.enabled(),
        log_file=log_file,
        checked_graph=discovery.checked_graph,
        config_cli=config_cli,
        config_base=config_base,
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
