"""Mutual exclusion among the run-time options AgL-running commands share."""

from __future__ import annotations

from agm.cli_support.args import ExecArgs


def log_option_conflict(*, no_log: bool, log: bool, log_file: str | None) -> str | None:
    """Return the usage error for mutually exclusive trace-logging options, if any."""
    if sum([no_log, log, log_file is not None]) > 1:
        return "--log, --no-log, and --log-file are mutually exclusive"
    return None


def exec_option_conflict(args: ExecArgs) -> str | None:
    """Return the usage error for mutually exclusive ``agm exec`` run-time options, if any."""
    for conflicting, first, second in (
        (args.log_file is not None and args.no_log_file, "--log-file", "--no-log-file"),
        (args.timeout is not None and args.no_timeout, "--timeout", "--no-timeout"),
    ):
        if conflicting:
            return f"{first} and {second} are mutually exclusive"
    return log_option_conflict(no_log=args.no_log, log=args.log, log_file=args.log_file)
