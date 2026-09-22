"""Mutual exclusion among the run-time options AgL-running commands share."""

from __future__ import annotations

from agm.cli_support.args import ExecArgs


def trace_option_conflict(*, no_trace: bool, trace: bool, trace_file: str | None) -> str | None:
    """Return the usage error for mutually exclusive trace-logging options, if any."""
    if sum([no_trace, trace, trace_file is not None]) > 1:
        return "--trace, --no-trace, and --trace-file are mutually exclusive"
    return None


def exec_option_conflict(args: ExecArgs) -> str | None:
    """Return the usage error for mutually exclusive ``agm exec`` run-time options, if any."""
    for conflicting, first, second in (
        (args.trace_file is not None and args.no_trace_file, "--trace-file", "--no-trace-file"),
        (args.timeout is not None and args.no_timeout, "--timeout", "--no-timeout"),
    ):
        if conflicting:
            return f"{first} and {second} are mutually exclusive"
    return trace_option_conflict(
        no_trace=args.no_trace, trace=args.trace, trace_file=args.trace_file
    )
