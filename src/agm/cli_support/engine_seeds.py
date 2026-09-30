"""Shared CLI-over-config resolution of the AgL engine settings a host controls.

``agm exec`` and ``agm repl`` both hand the AgL engine a seed mapping for the
settings the host explicitly controls.  The layering rule is the same for both —
CLI flag, then the command's configuration tables — so it lives here rather than
in either command.  :func:`build_host_engine_seeds` returns the three tiers as
an :class:`EngineSeedTiers`; :meth:`EngineSeedTiers.merged` flattens them, with
room for a caller-supplied middle tier, into the one mapping the engine seeds
from, and :meth:`EngineSeedTiers.trace_decision` reads the same resolution back
as the host's own trace-file decision, so a run's log file and its readable
``trace`` setting can never disagree. :func:`host_agent_spec_resolver` builds
the dispatch-time agent defaults both hosts apply from ``[agent]``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, cast

from agm.agent.spec_defaults import AgentSpecResolver, configured_defaults_resolver
from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
from agm.agl.runtime.engine_config import (
    raw_option_str,
    validate_engine_leaf_value,
    validate_engine_value,
)
from agm.agl.runtime.option import option_text
from agm.agl.semantics.values import BoolValue, RecordValue
from agm.config.engine_keys import (
    ENGINE_KEYS,
    EngineKeyKind,
    EngineKeySpec,
    trace_write_implies_enabled,
)
from agm.core.log import TraceDecision

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from agm.agl.semantics.type_table import TypeTable
    from agm.agl.semantics.values import Value
    from agm.cli_support.args import ExecutionOptionValues
    from agm.config.general import ExecConfig
    from agm.core.toml import TomlDict

__all__ = [
    "EngineSeedTiers",
    "build_host_engine_seeds",
    "execution_cli_values",
    "host_agent_spec_resolver",
    "resolve_timeout",
]


def execution_cli_values(args: "ExecutionOptionValues") -> dict[str, object | None]:
    """Return the engine settings *args*' execution flags explicitly set.

    A present ``None`` is an explicit empty ``Option`` (``--no-timeout``).
    """
    values: dict[str, object | None] = {}
    if args.strict_json is not None:
        values["strict-json"] = args.strict_json
    if args.timeout is not None:
        values["timeout"] = args.timeout
    elif args.no_timeout:
        values["timeout"] = None
    if args.no_trace:
        values["trace"] = False
    elif args.trace:
        values["trace"] = True
    if args.trace_file is not None:
        values["trace-file"] = args.trace_file
    if args.default_agent is not None:
        values["default-agent"] = args.default_agent
    if args.default_sandbox is not None:
        values["default-sandbox"] = args.default_sandbox
    if args.debug is not None:
        values["debug"] = args.debug
    if args.parse_error_retries is not None:
        values["parse-error-retries"] = args.parse_error_retries
    return values


def resolve_timeout(engine_seeds: "Mapping[str, Value]") -> float | None:
    """Parse the merged ``timeout`` seed into seconds; ``None`` when unset or cleared.

    Every tier's value is validated when decoded, so it always parses.
    """
    from agm.core.parse import parse_timeout

    seed = engine_seeds.get("timeout")
    if not isinstance(seed, RecordValue):
        return None
    text = option_text(seed, nominals=NO_BUILTIN_DECLARATIONS)
    return None if text is None else parse_timeout(text)


def resolve_strict_json(engine_seeds: "Mapping[str, Value]") -> bool:
    """Return the merged ``strict-json`` seed; ``False`` when unset."""
    seed = engine_seeds.get("strict-json")
    return isinstance(seed, BoolValue) and seed.value


def host_agent_spec_resolver(merged_config: "TomlDict") -> AgentSpecResolver:
    """Build the ``[agent]`` dispatch defaults; exit 1 naming the key when it is invalid."""
    try:
        return configured_defaults_resolver(merged_config)
    except ValueError as exc:
        print(f"Error: invalid agent configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _configured_value(
    spec: EngineKeySpec,
    config: "ExecConfig",
    primary_table: "Mapping[str, object]",
    fallback_table: "Mapping[str, object]",
) -> object | None:
    """Return one configured engine value, preserving raw ``Option[text]`` spelling."""
    assert spec.config_attr is not None
    configured_value = cast(object | None, getattr(config, spec.config_attr))
    if configured_value is None:
        return None
    if spec.kind is EngineKeyKind.OPTION_TEXT:
        return raw_option_str(primary_table, fallback_table, spec.name)
    return configured_value


def _exit_on_invalid[T](key_name: str, origin: str, check: "Callable[[], T]") -> T:
    """Run *check*; on ``ValueError`` print an origin-tagged error and exit 1."""
    try:
        return check()
    except ValueError as exc:
        print(f"Error: invalid {key_name} value from {origin}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _decode_engine_value(
    key_name: str, raw: object, origin: str, type_table: "TypeTable"
) -> "Value":
    """Decode one engine value, printing an origin-tagged error and exiting 1 on failure.

    Shared by the ordinary per-key seeding loop and the derived ``trace`` rule,
    so both go through one decode-failure contract (message and exit code).
    Decoding — including ``timeout``'s duration-syntax check — is
    :func:`~agm.agl.runtime.engine_config.validate_engine_leaf_value`'s rule,
    so a CLI flag, a config-file entry, and a package manifest leaf fail
    identically.
    """
    return _exit_on_invalid(
        key_name, origin, partial(validate_engine_leaf_value, key_name, raw, type_table)
    )


def validate_config_engine_values(values: "Mapping[str, Value]") -> None:
    """Validate every decoded ``@config`` engine value, exiting 1 naming the first invalid key.

    Applies :func:`~agm.agl.runtime.engine_config.validate_engine_value`
    whether or not a higher tier overrides the value, like a CLI or
    config-table value.
    """
    for name, value in values.items():
        _exit_on_invalid(name, "@config", partial(validate_engine_value, name, value))


@dataclass(frozen=True)
class EngineSeedTiers:
    """``cli`` (explicit CLI flags) above ``upper`` (primary table) above ``lower`` (fallback).

    A config value the CLI overrides is absent from ``upper``/``lower``.
    ``cli_trace`` is the derived ``trace`` setting an explicit CLI flag fixes.
    """

    cli: "Mapping[str, Value]"
    upper: "Mapping[str, Value]"
    lower: "Mapping[str, Value]"
    cli_trace: "tuple[object, str] | None"
    _type_table: "TypeTable" = field(repr=False)

    def config_merged(self, middle: "Mapping[str, Value] | None" = None) -> "dict[str, Value]":
        """Flatten the config-only tiers: ``lower`` < *middle* < ``upper``, ``cli`` excluded.

        *middle*: already-decoded engine settings ranked between ``lower``
        and ``upper`` (a selected program's own ``@config`` entries, restamped
        onto the standard identity by the caller). Exposed so a host reads
        the same config-only view :meth:`merged` derives ``trace`` from — e.g.
        to resolve its own trace-file decision — without recomputing it.
        """
        mid: "Mapping[str, Value]" = middle if middle is not None else {}
        return {**self.lower, **mid, **self.upper}

    def merged(self, middle: "Mapping[str, Value] | None" = None) -> "dict[str, Value]":
        """Flatten to one mapping: ``lower`` < *middle* < ``upper`` < ``cli``, plus ``trace``.

        *middle*: already-decoded engine settings ranked between ``lower``
        and ``upper``. The derived ``trace`` setting is resolved from the
        config-only merge (``cli`` excluded, except for its own fast path) —
        see :meth:`_resolve_trace`.
        """
        config_result = self.config_merged(middle)
        result = {**config_result, **self.cli}
        trace = self._resolve_trace(config_result)
        if trace is not None:
            result["trace"] = trace
        return result

    def trace_decision(self, middle: "Mapping[str, Value] | None" = None) -> TraceDecision:
        """Resolve the trace decision the host's own log file starts from.

        The same resolution :meth:`merged` seeds ``trace`` with, read back as
        an on/off state plus the explicit destination (``None`` → the
        auto-generated timestamped path).  Left unconfigured everywhere,
        tracing is off.  The path is the CLI's when it named one, otherwise
        the config tiers'.

        A program's own ``std/config::trace``/``trace-file`` write is not a
        layer here: it takes effect at runtime, from its program point onward,
        through the host settings reconfigurer, and overrides this decision —
        see ``docs/agl/reference/host-environment.md``.
        """
        from agm.agl.semantics.values import BoolValue

        config_result = self.config_merged(middle)
        trace = self._resolve_trace(config_result)
        path = self._trace_file_text(self.cli)
        if path is None:
            path = self._trace_file_text(config_result)
        return TraceDecision(
            enabled=isinstance(trace, BoolValue) and trace.value,
            explicit_path=path,
        )

    @staticmethod
    def _trace_file_text(values: "Mapping[str, Value]") -> str | None:
        """Return the trace path *values* carries, or ``None`` for an empty/absent one."""
        from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
        from agm.agl.runtime.option import option_text
        from agm.agl.semantics.values import RecordValue

        value = values.get("trace-file")
        if not isinstance(value, RecordValue):
            return None
        return option_text(value, nominals=NO_BUILTIN_DECLARATIONS)

    def _resolve_trace(self, config_result: "Mapping[str, Value]") -> "Value | None":
        """Recompute the derived ``trace`` setting.

        An explicit CLI ``trace`` or ``trace-file`` wins outright. Otherwise
        ``trace`` is derived from *config_result* alone
        (``lower``/*middle*/``upper``, ``cli`` excluded): when it names
        ``trace`` or ``trace-file`` at all, ``trace`` is true iff its ``trace``
        is true or its ``trace-file`` resolves to a real path (``Some``) — each
        key independently carrying whichever tier won that merge. Otherwise an
        effective ``debug`` (CLI, else config) turns ``trace`` on. Left
        unconfigured everywhere, ``trace`` stays absent (``None``).
        """
        from agm.agl.semantics.values import BoolValue

        debug = self.cli.get("debug", config_result.get("debug"))
        if self.cli_trace is not None:
            raw, origin = self.cli_trace
        elif "trace" in config_result or "trace-file" in config_result:
            trace_value = config_result.get("trace")
            is_trace_true = isinstance(trace_value, BoolValue) and trace_value.value
            # The config layer follows the declared register relation a
            # runtime write follows: a real destination implies the switch.
            raw = is_trace_true or trace_write_implies_enabled(
                "trace-file", self._trace_file_text(config_result) is not None
            )
            origin = "trace/trace-file configuration"
        elif isinstance(debug, BoolValue) and debug.value:
            raw, origin = True, "debug"
        else:
            return None

        return _decode_engine_value("trace", raw, origin, self._type_table)


def build_host_engine_seeds(
    *,
    config: "ExecConfig",
    primary_table: "Mapping[str, object]",
    fallback_table: "Mapping[str, object] | None" = None,
    cli_values: "Mapping[str, object | None]",
) -> "EngineSeedTiers":
    """Decode the engine settings the host explicitly controls into a three-tier seed set.

    ``cli`` holds every explicitly supplied CLI value; ``upper`` holds every
    *primary_table* value; ``lower`` holds every *fallback_table* value not
    already covered by *primary_table*. A table value the CLI overrides is
    not decoded. A setting left to its default in every source stays absent
    from every tier, so a ``builtin var`` initializer supplies it instead of
    being suppressed by a host-side floor. ``cli_values`` contains only
    explicitly supplied CLI values; a present ``None`` is an explicit empty
    ``Option``. ``trace`` is seeded like any other key here;
    :meth:`EngineSeedTiers.merged` recomputes it once the ``trace-file``
    implication and any caller-supplied middle tier are folded in.

    Every key, ``default-agent`` included, decodes through
    :func:`~agm.agl.runtime.engine_config.validate_engine_leaf_value` against
    one shared seeded ``TypeTable``. A decode failure prints an error naming
    the offending key and its origin (``--<name>`` for a CLI flag, otherwise
    ``configuration key <name>``) to stderr and exits 1 here, before anything
    runs.
    """
    from agm.agl.semantics.type_table import create_seeded_type_table

    fallback: "Mapping[str, object]" = fallback_table if fallback_table is not None else {}
    type_table = create_seeded_type_table()

    cli: dict[str, Value] = {}
    upper: dict[str, Value] = {}
    lower: dict[str, Value] = {}
    for spec in ENGINE_KEYS:
        if spec.name in cli_values:
            cli[spec.name] = _decode_engine_value(
                spec.name, cli_values[spec.name], f"--{spec.name}", type_table
            )
            continue
        if spec.name in primary_table:
            tier = upper
        elif spec.name in fallback:
            tier = lower
        else:
            continue
        value = _configured_value(spec, config, primary_table, fallback)
        # A ``None`` config result is absent, not an explicit control; this
        # lets a builtin initializer supply invalid/empty Option values.
        if value is not None:
            tier[spec.name] = _decode_engine_value(
                spec.name, value, f"configuration key {spec.name}", type_table
            )

    if "trace" in cli_values:
        cli_trace: tuple[object, str] | None = (cli_values["trace"], "--trace")
    elif "trace-file" in cli_values:
        cli_trace = (True, "--trace-file")
    else:
        cli_trace = None

    return EngineSeedTiers(
        cli=cli, upper=upper, lower=lower, cli_trace=cli_trace, _type_table=type_table
    )
