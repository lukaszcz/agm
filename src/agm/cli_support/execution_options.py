"""Shared declarations and help display for AgL execution options."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import click

#: Widest option label kept on its description's line, at most half the help width.
_LABEL_COLUMN_MAX = 34


@dataclass(frozen=True, slots=True)
class ExecutionOptionSpec:
    """One shared execution option's Click declaration and help grouping."""

    name: str
    declarations: tuple[str, ...]
    metavar: str | None
    group: str


@dataclass(frozen=True, slots=True)
class _HelpGroup:
    """One help row: the grouped options' labels, joined by `` / ``, and their help."""

    key: str
    description: str


EXECUTION_OPTION_SPECS: tuple[ExecutionOptionSpec, ...] = (
    ExecutionOptionSpec("strict_json", ("--strict-json/--no-strict-json",), None, "strict_json"),
    ExecutionOptionSpec("max_call_depth", ("--max-call-depth",), "N", "call_depth"),
    ExecutionOptionSpec("default_agent", ("--default-agent",), "AGENT", "default_agent"),
    ExecutionOptionSpec("default_sandbox", ("--default-sandbox",), "SANDBOX", "default_sandbox"),
    ExecutionOptionSpec("timeout", ("--timeout",), "DURATION", "timeout"),
    ExecutionOptionSpec("no_timeout", ("--no-timeout",), None, "timeout"),
    ExecutionOptionSpec("trace", ("--trace",), None, "trace"),
    ExecutionOptionSpec("no_trace", ("--no-trace",), None, "trace"),
    ExecutionOptionSpec("trace_file", ("--trace-file",), "PATH", "trace_file"),
    ExecutionOptionSpec("debug", ("--debug/--no-debug",), None, "debug"),
    ExecutionOptionSpec(
        "parse_error_retries", ("--parse-error-retries",), "N", "parse_error_retries"
    ),
)

_SPEC_BY_NAME = {spec.name: spec for spec in EXECUTION_OPTION_SPECS}
#: Every execution option's parameter name; ``agm exec``, ``agm repl``, and
#: registered commands all accept each one.
EXECUTION_OPTION_NAMES: frozenset[str] = frozenset(_SPEC_BY_NAME)
_HELP_GROUPS: tuple[_HelpGroup, ...] = (
    _HelpGroup("strict_json", "Strict or lenient agent JSON parsing."),
    _HelpGroup("call_depth", "Maximum call depth."),
    _HelpGroup("default_agent", "Default agent (Agent value or command)."),
    _HelpGroup("default_sandbox", "Default agent sandbox (AgentSandbox)."),
    _HelpGroup("timeout", "Shell-exec/agent idle timeout, or none."),
    _HelpGroup("trace", "Enable or disable trace logging."),
    _HelpGroup("trace_file", "Write the JSONL trace to PATH."),
    _HelpGroup("debug", "Debug mode."),
    _HelpGroup("parse_error_retries", "Corrective retry count used by ask's default parse policy."),
)


def execution_option_spec(name: str) -> ExecutionOptionSpec:
    """Return the shared declaration metadata for one execution option."""
    return _SPEC_BY_NAME[name]


def execution_option_rows(parameter_names: Collection[str]) -> tuple[tuple[str, str], ...]:
    """Return ``(options, help)`` rows for the given supported execution options."""
    names = frozenset(parameter_names)
    rows: list[tuple[str, str]] = []
    for group in _HELP_GROUPS:
        specs = [
            spec
            for spec in EXECUTION_OPTION_SPECS
            if spec.group == group.key and spec.name in names
        ]
        if not specs:
            continue
        labels = [
            " ".join(filter(None, (spec.declarations[0].replace("/", " / "), spec.metavar)))
            for spec in specs
        ]
        rows.append((" / ".join(labels), group.description))
    return tuple(rows)


def write_execution_options(
    formatter: click.HelpFormatter, parameter_names: Collection[str]
) -> None:
    """Write the ``Execution options`` section for *parameter_names*, if any, to *formatter*."""
    rows = execution_option_rows(parameter_names)
    if rows:
        with formatter.section("Execution options"):
            formatter.write_dl(rows, col_max=min(_LABEL_COLUMN_MAX, formatter.width // 2))


def execution_options_section(*, width: int) -> str:
    """Render the ``Execution options`` section, wrapped to *width*."""
    formatter = click.HelpFormatter(width=width)
    write_execution_options(formatter, EXECUTION_OPTION_NAMES)
    return formatter.getvalue()
