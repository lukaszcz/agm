"""Shared CLI-over-config resolution of the AgL engine settings a host controls.

``agm exec`` and ``agm repl`` both hand the AgL engine a seed mapping for the
settings the host explicitly controls.  The layering rule is the same for both —
CLI flag, then the command's configuration tables — so it lives here rather than
in either command.  :func:`build_host_engine_seeds` returns the three tiers as
an :class:`EngineSeedTiers`; :meth:`EngineSeedTiers.merged` flattens them, with
room for a caller-supplied middle tier, into the one mapping the engine seeds
from.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar, cast

from agm.agl.runtime.engine_config import convert_config_value, raw_option_str
from agm.config.engine_keys import ENGINE_KEYS, EngineKeyKind, EngineKeySpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agm.agl.semantics.type_table import TypeTable
    from agm.agl.semantics.values import Value
    from agm.config.general import ExecConfig

__all__ = ["EngineSeedTiers", "build_host_engine_seeds"]

_T = TypeVar("_T")


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


def _validate_cli_timeout(raw: str) -> None:
    """Reject an unparsable ``--timeout`` value eagerly, before anything runs.

    ``convert_config_value`` only checks ``--timeout``'s value decodes as
    ``Option[text]``, never that it is a valid duration — that conversion
    happens once at its use site (:func:`~agm.core.parse.parse_timeout`).
    Validating it here, alongside every other CLI decode failure, keeps a
    malformed flag from surfacing only after the static pipeline has already
    run.
    """
    from agm.core.parse import parse_timeout

    try:
        parse_timeout(raw)
    except ValueError as exc:
        print(f"Error: invalid --timeout value: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _decode_engine_value(
    key_name: str, raw: object, origin: str, type_table: "TypeTable"
) -> "Value":
    """Decode one engine value, printing an origin-tagged error and exiting 1 on failure.

    Shared by the ordinary per-key seeding loop and the derived ``trace`` rule,
    so both go through one decode-failure contract (message and exit code).
    """
    from agm.agl.semantics.engine_keys import ENGINE_KEY_TYPES

    try:
        return convert_config_value(key_name, raw, ENGINE_KEY_TYPES[key_name], type_table)
    except ValueError as exc:
        print(f"Error: invalid {key_name} value from {origin}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


@dataclass(frozen=True)
class EngineSeedTiers:
    """``cli`` (explicit CLI flags) above ``upper`` (primary table) above ``lower`` (fallback).

    The three tiers are independent: ``upper``/``lower`` always reflect the
    config tables, whether or not ``cli`` also names the same key — this
    matters for ``trace-file``, where ``--no-trace-file`` clears only the CLI
    seed and must not hide a config-table path from the derived ``trace`` rule
    (see :meth:`merged`). ``cli`` still wins the actual seeded value for a
    key it names.
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

    def _resolve_trace(self, config_result: "Mapping[str, Value]") -> "Value | None":
        """Recompute the derived ``trace`` setting.

        An explicit CLI ``trace``, or a non-``None`` CLI ``trace-file``, wins
        outright. Otherwise ``trace`` is derived from *config_result* alone
        (``lower``/*middle*/``upper``, ``cli`` excluded): when it names
        ``trace`` or ``trace-file`` at all, ``trace`` is true iff its ``trace``
        is true or its ``trace-file`` resolves to a real path (``Some``) — each
        key independently carrying whichever tier won that merge. Left
        unconfigured everywhere, ``trace`` stays absent (``None``).

        Excluding ``cli`` here (beyond its own fast path) is deliberate:
        ``--no-trace-file`` clears only the CLI seed, not a trace a config
        table independently establishes.
        """
        if self.cli_trace is not None:
            raw, origin = self.cli_trace
        elif "trace" in config_result or "trace-file" in config_result:
            from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
            from agm.agl.runtime.option import option_text
            from agm.agl.semantics.values import BoolValue, RecordValue

            trace_value = config_result.get("trace")
            is_trace_true = isinstance(trace_value, BoolValue) and trace_value.value
            trace_file_value = config_result.get("trace-file")
            trace_file_path = (
                option_text(trace_file_value, nominals=NO_BUILTIN_DECLARATIONS)
                if isinstance(trace_file_value, RecordValue)
                else None
            )
            raw = is_trace_true or trace_file_path is not None
            origin = "trace/trace-file configuration"
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
    not decoded, except ``trace-file``, which the derived ``trace`` rule still
    reads from the config tiers. A setting
    left to its default in every source stays absent from every tier, so a
    ``builtin var`` initializer supplies it instead of being suppressed by a
    host-side floor. ``cli_values`` contains only explicitly supplied CLI
    values; its present ``None`` values represent an explicit empty
    ``Option``. ``trace`` is seeded like any other key here;
    :meth:`EngineSeedTiers.merged` recomputes it once the ``trace-file``
    implication and any caller-supplied middle tier are folded in.

    Every key, ``default-agent`` included, decodes through
    :func:`~agm.agl.runtime.engine_config.convert_config_value` against one
    shared seeded ``TypeTable``. A decode failure prints an error naming the
    offending key and its origin (``--<name>`` for a CLI flag, otherwise
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
            if spec.name == "timeout" and cli_values["timeout"] is not None:
                _validate_cli_timeout(cast(str, cli_values["timeout"]))
            cli[spec.name] = _decode_engine_value(
                spec.name, cli_values[spec.name], f"--{spec.name}", type_table
            )
            # An overridden table value is never decoded, except ``trace-file``:
            # ``--no-trace-file`` leaves a configured trace path enabling ``trace``.
            if spec.name != "trace-file":
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
    elif cli_values.get("trace-file") is not None:
        cli_trace = (True, "--trace-file")
    else:
        cli_trace = None

    return EngineSeedTiers(
        cli=cli, upper=upper, lower=lower, cli_trace=cli_trace, _type_table=type_table
    )
