"""Implementation of the ``agm exec FILE`` command.

M0 behaviour: reads the ``.agl`` source file (exit 1 if unreadable), constructs
a ``WorkflowRuntime``, calls ``runtime.run``, prints diagnostics to stderr, and
exits 1 (pre-execution failure per the exit-code contract, since the runtime is
not yet implemented in M0).

Warning-severity diagnostics are printed to stderr like errors but never affect
the exit code; only error-severity diagnostics yield exit 1.

Exit-code contract (plan §10.1):
    0  success
    1  pre-execution failure (unreadable file, static errors, input validation)
    2  program executed but ended with an uncaught AgL exception
"""

from __future__ import annotations

import sys
from pathlib import Path

from agm.agl import WorkflowRuntime
from agm.commands.args import ExecArgs
from agm.core.cli_helpers import parse_inputs
from agm.core.fs import read_text_arg


def run(args: ExecArgs) -> None:
    """Run the ``agm exec`` command."""
    source = read_text_arg(Path(args.file))

    # Parse --input k=v pairs (validation: malformed pairs exit 1).
    try:
        inputs = parse_inputs(args.inputs)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # Construct the runtime with CLI/config-derived settings.
    # In M0 the config is not yet wired; flags default to sensible values.
    strict_json = args.strict_json if args.strict_json is not None else False
    loop_limit = args.max_iters if args.max_iters is not None else 5

    runtime = WorkflowRuntime(
        default_loop_limit=loop_limit,
        default_strict_json=strict_json,
    )

    # ``parse_inputs`` returns ``dict[str, str]``; ``run`` accepts a
    # ``Mapping[str, object]``, so no widening copy is needed.
    result = runtime.run(source, inputs=inputs)

    # Warnings are reported but do not affect the exit code.
    warnings = [d for d in result.diagnostics if d.severity == "warning"]
    errors = [d for d in result.diagnostics if d.severity == "error"]

    for diag in warnings:
        print(f"line {diag.line}: {diag.message}", file=sys.stderr)

    if result.ok:
        return

    # Pre-execution failure: print error diagnostics and exit 1.
    if result.error is None:
        for diag in errors:
            print(f"line {diag.line}: {diag.message}", file=sys.stderr)
        raise SystemExit(1)

    # Uncaught AgL exception: print and exit 2.
    message = result.error.fields.get("message")
    if isinstance(message, str) and message:
        print(f"AgL exception: {result.error.type_name}: {message}", file=sys.stderr)
    else:
        print(f"AgL exception: {result.error.type_name}", file=sys.stderr)
    raise SystemExit(2)
