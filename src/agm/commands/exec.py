"""Implementation of the ``agm exec FILE`` command.

Behaviour: read the ``.agl`` source file (exit 1 if unreadable), load the
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
    1  pre-execution failure (unreadable file, static errors, input validation)
    2  program executed but ended with an uncaught AgL exception

Flag notes:
    - ``--strict-json`` controls JSON-codec strictness: when set, agents must
      return exactly one bare JSON value; the default is lenient recovery
      (fence/prose stripping + trivial repair, then strict schema validation).
      A source-level ``strict_json`` call option overrides this default.
    - ``--log-file PATH`` writes a structured JSONL trace under the given path.
    - ``--no-log`` disables trace logging entirely.
    - ``--runner`` is accepted but inert until the runner-backed default agent
      lands in M5.
    - ``--dry-run`` (global flag) runs only the static pipeline + contract
      materialization and never writes a trace (side-effect-free).
"""

from __future__ import annotations

import sys
from pathlib import Path

from agm.agl import WorkflowRuntime
from agm.commands.args import ExecArgs
from agm.config.context import current_config_context
from agm.config.general import load_exec_config
from agm.core import dry_run
from agm.core.cli_helpers import parse_inputs
from agm.core.fs import read_text_arg
from agm.core.log import resolve_log_file


def run(args: ExecArgs) -> None:
    """Run the ``agm exec`` command."""
    source = read_text_arg(Path(args.file))

    # Parse --input k=v pairs (validation: malformed pairs exit 1).
    try:
        inputs = parse_inputs(args.inputs)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # Load [exec] configuration; CLI flags override config values.
    ctx = current_config_context()
    config = load_exec_config(home=ctx.home, proj_dir=ctx.proj_dir, cwd=ctx.cwd)

    strict_json = args.strict_json if args.strict_json is not None else config.strict_json
    loop_limit = args.max_iters if args.max_iters is not None else config.default_loop_limit

    # Resolve the trace log file.  --dry-run is side-effect-free: no trace
    # is written regardless of --log-file (plan §10.1).
    if dry_run.enabled():
        log_file = None
    else:
        log_file = resolve_log_file(
            command_name="exec",
            no_log=args.no_log,
            log_file=args.log_file,
        )

    runtime = WorkflowRuntime(
        default_loop_limit=loop_limit,
        default_strict_json=strict_json,
        shell_exec_timeout=config.timeout,
    )

    # ``parse_inputs`` returns ``dict[str, str]``; ``run`` accepts a
    # ``Mapping[str, object]``, so no widening copy is needed.
    result = runtime.run(
        source,
        inputs=inputs,
        check_only=dry_run.enabled(),
        log_file=log_file,
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

    # Uncaught AgL exception: print and exit 2.
    message = result.error.fields.get("message")
    if isinstance(message, str) and message:
        print(f"AgL exception: {result.error.type_name}: {message}", file=sys.stderr)
    else:
        print(f"AgL exception: {result.error.type_name}", file=sys.stderr)
    raise SystemExit(2)
