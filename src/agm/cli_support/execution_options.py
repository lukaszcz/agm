"""Shared declarations and help display for AgL execution options."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

ExecutionSurface = Literal["exec", "repl", "registered"]


@dataclass(frozen=True, slots=True)
class ExecutionOptionSpec:
    """One shared execution option's Click declaration and help grouping."""

    name: str
    declarations: tuple[str, ...]
    metavar: str | None
    group: str
    surfaces: frozenset[ExecutionSurface]


_EXEC_AND_REGISTERED: frozenset[ExecutionSurface] = frozenset({"exec", "registered"})
_ALL_SURFACES: frozenset[ExecutionSurface] = frozenset({"exec", "repl", "registered"})

EXECUTION_OPTION_SPECS: tuple[ExecutionOptionSpec, ...] = (
    ExecutionOptionSpec(
        "strict_json", ("--strict-json/--no-strict-json",), None, "strict_json", _ALL_SURFACES
    ),
    ExecutionOptionSpec("max_call_depth", ("--max-call-depth",), "N", "call_depth", _ALL_SURFACES),
    ExecutionOptionSpec(
        "default_agent", ("--default-agent",), "AGENT", "default_agent", _ALL_SURFACES
    ),
    ExecutionOptionSpec(
        "default_sandbox", ("--default-sandbox",), "SANDBOX", "default_sandbox", _ALL_SURFACES
    ),
    ExecutionOptionSpec("timeout", ("--timeout",), "DURATION", "timeout", _EXEC_AND_REGISTERED),
    ExecutionOptionSpec("no_timeout", ("--no-timeout",), None, "timeout", _EXEC_AND_REGISTERED),
    ExecutionOptionSpec("trace", ("--trace",), None, "trace", _ALL_SURFACES),
    ExecutionOptionSpec("no_trace", ("--no-trace",), None, "trace", _ALL_SURFACES),
    ExecutionOptionSpec("trace_file", ("--trace-file",), "PATH", "trace", _ALL_SURFACES),
    ExecutionOptionSpec("no_trace_file", ("--no-trace-file",), None, "trace", _EXEC_AND_REGISTERED),
)

_SPEC_BY_NAME = {spec.name: spec for spec in EXECUTION_OPTION_SPECS}
_HELP_GROUPS: tuple[tuple[str, str], ...] = (
    ("strict_json", " "),
    ("call_depth", " "),
    ("default_agent", " "),
    ("default_sandbox", " "),
    ("timeout", " / "),
    ("trace", ", "),
)


def execution_option_spec(name: str) -> ExecutionOptionSpec:
    """Return the shared declaration metadata for one execution option."""
    return _SPEC_BY_NAME[name]


def execution_option_lines(parameter_names: Collection[str]) -> tuple[str, ...]:
    """Format supported execution options as concise, grouped help rows."""
    names = frozenset(parameter_names)
    rows: list[str] = []
    for group, separator in _HELP_GROUPS:
        specs = [
            spec for spec in EXECUTION_OPTION_SPECS if spec.group == group and spec.name in names
        ]
        if not specs:
            continue
        labels = [
            f"{spec.declarations[0].replace('/', ' / ')} {spec.metavar}"
            if spec.metavar is not None
            else spec.declarations[0].replace("/", " / ")
            for spec in specs
        ]
        rows.append(separator.join(labels))
    return tuple(rows)


def execution_options_section(surface: ExecutionSurface) -> str:
    """Render the final help section for a built-in execution surface."""
    return format_execution_options_section(execution_option_names_for_surface(surface))


def execution_option_names_for_surface(surface: ExecutionSurface) -> frozenset[str]:
    """Return the shared execution option parameters supported by *surface*."""
    return frozenset(spec.name for spec in EXECUTION_OPTION_SPECS if surface in spec.surfaces)


def format_execution_options_section(parameter_names: Collection[str]) -> str:
    """Render the final help section for the given execution-option parameters."""
    return _format_execution_options_section(execution_option_lines(parameter_names))


def _format_execution_options_section(rows: tuple[str, ...]) -> str:
    if not rows:
        return ""
    return "Execution options:\n" + "".join(f"  {row}\n" for row in rows)
