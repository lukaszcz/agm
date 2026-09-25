"""Mutual exclusion among the run-time options AgL-running commands share.

The exclusive groups are derived from the engine-key catalog through the same
CLI projection that generates the flags, so a key added there brings its own
rule with it instead of needing a row here.
"""

from __future__ import annotations

from agm.cli_support.args import ExecutionOptionValues


def _exclusive_flag_groups() -> tuple[tuple[str, ...], ...]:
    """Return the engine-key flag groups of which at most one may be supplied.

    Keys sharing a register come first: together they name one destination, so
    ``--trace-file`` excludes ``--trace``/``--no-trace``.  A key that only
    *implies* the register is enabled keeps its own negative out of that group,
    since clearing it does not turn the register off.  Every key then excludes
    the negative the projection generates for it (``--timeout`` /
    ``--no-timeout``).
    """
    from agm.agl.semantics.engine_keys import ENGINE_KEY_TYPES
    from agm.cli_support.program_options import project_option
    from agm.config.engine_keys import ENGINE_KEYS, ENGINE_REGISTERS

    projected = {
        spec.name: project_option(spec.name, ENGINE_KEY_TYPES[spec.name]) for spec in ENGINE_KEYS
    }
    groups: list[tuple[str, ...]] = []
    for register in ENGINE_REGISTERS:
        group: list[str] = []
        for spec in ENGINE_KEYS:
            if spec.register != register:
                continue
            option = projected[spec.name]
            group.append(option.flag)
            if not spec.enables_register and option.negative_flag is not None:
                group.append(option.negative_flag)
        groups.append(tuple(group))
    groups.extend(
        (option.flag, option.negative_flag)
        for option in projected.values()
        if option.negative_flag is not None
    )
    return tuple(groups)


def _join_flags(flags: tuple[str, ...]) -> str:
    if len(flags) == 2:
        return f"{flags[0]} and {flags[1]}"
    return f"{', '.join(flags[:-1])}, and {flags[-1]}"


def _flag_conflict(supplied: set[str]) -> str | None:
    """Return the usage error for the first group *supplied* names twice, if any."""
    for group in _exclusive_flag_groups():
        if len(supplied.intersection(group)) > 1:
            return f"{_join_flags(group)} are mutually exclusive"
    return None


def execution_option_conflict(args: ExecutionOptionValues) -> str | None:
    """Return the usage error for mutually exclusive execution options, if any."""
    supplied = {
        flag
        for flag, given in (
            ("--trace", args.trace),
            ("--no-trace", args.no_trace),
            ("--trace-file", args.trace_file is not None),
            ("--no-trace-file", args.no_trace_file),
            ("--timeout", args.timeout is not None),
            ("--no-timeout", args.no_timeout),
        )
        if given
    }
    return _flag_conflict(supplied)
