"""Shared CLI-over-config resolution of the AgL engine settings a host controls.

``agm exec`` and ``agm repl`` both hand the AgL engine a seed mapping for the
settings the host explicitly controls.  The layering rule is the same for both —
CLI flag, then the command's configuration tables — so it lives here rather than
in either command.  :func:`build_host_engine_seeds` returns the two tiers as an
:class:`EngineSeedTiers`; :meth:`EngineSeedTiers.merged` flattens them, with
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


def _decode_engine_value(
    key_name: str, raw: object, origin: str, type_table: "TypeTable"
) -> "Value":
    """Decode one engine value, printing an origin-tagged error and exiting 1 on failure.

    Shared by the ordinary per-key seeding loop and the derived ``log`` rule,
    so both go through one decode-failure contract (message and exit code).
    """
    from agm.agl.semantics.engine_keys import ENGINE_KEY_TYPES

    try:
        return convert_config_value(key_name, raw, ENGINE_KEY_TYPES[key_name], type_table)
    except ValueError as exc:
        print(f"Error: invalid {key_name} value from {origin}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


@dataclass(frozen=True)
class _LogPresence:
    """Whether one raw config table sets ``log``/``log-file``, value validity aside.

    Membership only: a tier "sets" a key by naming it at all, even with an
    invalid or empty value — the same gate the per-key resolution uses.
    """

    has_log: bool
    has_log_file: bool


def _table_log_presence(table: "Mapping[str, object]") -> _LogPresence:
    """Return which of ``log``/``log-file`` one raw config table names."""
    return _LogPresence(has_log="log" in table, has_log_file="log-file" in table)


@dataclass(frozen=True)
class _MiddleLogState:
    """The middle tier's ``log``/``log-file``, read from already-decoded engine ``Value``s."""

    has_log: bool
    log: bool
    has_log_file: bool
    log_file: str | None


def _middle_log_state(middle: "Mapping[str, Value]") -> _MiddleLogState:
    """Read ``log``/``log-file`` from the middle tier's decoded ``Value``s.

    A ``log-file`` decoded as ``Some(text)`` counts as a real path whatever
    that text is — like a CLI ``--log-file`` value, it is never re-validated
    here. A ``log-file`` decoded as ``None`` still counts as *set*: it
    explicitly hides a lower-tier ``log-file`` instead of falling through to
    one, the same way a raw table naming ``log-file`` with an invalid value
    still shadows a lower tier's value (see :class:`_LogPresence`).
    """
    from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
    from agm.agl.runtime.option import option_text
    from agm.agl.semantics.values import BoolValue, RecordValue

    has_log = "log" in middle
    log = False
    if has_log:
        log_value = middle["log"]
        assert isinstance(log_value, BoolValue)
        log = log_value.value

    has_log_file = "log-file" in middle
    log_file = None
    if has_log_file:
        log_file_value = middle["log-file"]
        assert isinstance(log_file_value, RecordValue)
        log_file = option_text(log_file_value, nominals=NO_BUILTIN_DECLARATIONS)

    return _MiddleLogState(has_log=has_log, log=log, has_log_file=has_log_file, log_file=log_file)


def _highest_tier(
    *,
    upper_has: bool,
    middle_has: bool,
    middle_value: _T,
    lower_has: bool,
    config_value: _T,
    default: _T,
) -> _T:
    """Pick the value from whichever of (upper, middle, lower) sets it, in that order.

    ``config_value`` serves both ``upper`` and ``lower``: ``ExecConfig``
    already folds the primary table over the fallback table for ``log`` and
    ``log-file``, so it equals whichever of those two raw tables actually
    sets the key — the middle tier is the only one needing its own value.
    """
    if upper_has:
        return config_value
    if middle_has:
        return middle_value
    if lower_has:
        return config_value
    return default


@dataclass(frozen=True)
class _LogSeedInputs:
    """Raw inputs :meth:`EngineSeedTiers.merged` needs to recompute the derived ``log`` key.

    ``cli`` holds ``(native_value, origin)`` when an explicit CLI ``log`` or
    ``log-file`` already decides the setting, bypassing every tier below.
    """

    cli: tuple[object, str] | None
    upper: _LogPresence
    lower: _LogPresence
    config_log: bool
    config_log_file: str | None
    type_table: "TypeTable"


@dataclass(frozen=True)
class EngineSeedTiers:
    """``upper`` (CLI, primary table) above ``lower`` (fallback table).

    :meth:`merged` layers an optional caller-supplied *middle* tier between
    them. The derived ``log`` setting — never a member of ``upper``/``lower``
    — is recomputed once after that layering; see :meth:`merged`.
    """

    upper: "Mapping[str, Value]"
    lower: "Mapping[str, Value]"
    log_inputs: _LogSeedInputs = field(repr=False)

    def merged(self, middle: "Mapping[str, Value] | None" = None) -> "dict[str, Value]":
        """Flatten to one seed mapping: ``lower`` < *middle* < ``upper``, plus derived ``log``.

        *middle*: already-decoded engine settings ranked between ``lower``
        and ``upper``.
        """
        mid: "Mapping[str, Value]" = middle if middle is not None else {}
        result = {**self.lower, **mid, **self.upper}
        log = self._resolve_log(mid)
        if log is not None:
            result["log"] = log
        return result

    def _resolve_log(self, middle: "Mapping[str, Value]") -> "Value | None":
        """Recompute the derived ``log`` setting across CLI, *middle*, and the two tables.

        An explicit CLI ``log`` wins; else a CLI ``log-file`` implies
        ``log = True``; else ``log`` and ``log-file`` each resolve
        independently from whichever of the program table, *middle*, or
        ``[exec]`` (checked in that order) sets them, and ``log`` is that
        ``log`` value ORed with whether that ``log-file`` gave it a real
        path. Left to its default everywhere, ``log`` stays absent (``None``).
        """
        if self.log_inputs.cli is not None:
            raw, origin = self.log_inputs.cli
        else:
            mid_log = _middle_log_state(middle)
            upper, lower = self.log_inputs.upper, self.log_inputs.lower
            if not (
                upper.has_log
                or upper.has_log_file
                or mid_log.has_log
                or mid_log.has_log_file
                or lower.has_log
                or lower.has_log_file
            ):
                return None
            merged_log = _highest_tier(
                upper_has=upper.has_log,
                middle_has=mid_log.has_log,
                middle_value=mid_log.log,
                lower_has=lower.has_log,
                config_value=self.log_inputs.config_log,
                default=False,
            )
            merged_log_file = _highest_tier(
                upper_has=upper.has_log_file,
                middle_has=mid_log.has_log_file,
                middle_value=mid_log.log_file,
                lower_has=lower.has_log_file,
                config_value=self.log_inputs.config_log_file,
                default=None,
            )
            raw = merged_log or merged_log_file is not None
            origin = "log/log-file configuration"

        return _decode_engine_value("log", raw, origin, self.log_inputs.type_table)


def build_host_engine_seeds(
    *,
    config: "ExecConfig",
    primary_table: "Mapping[str, object]",
    fallback_table: "Mapping[str, object] | None" = None,
    cli_values: "Mapping[str, object | None]",
) -> "EngineSeedTiers":
    """Decode the engine settings the host explicitly controls into a two-tier seed set.

    ``upper`` holds every explicitly supplied CLI value and every
    *primary_table* value not already covered by one; ``lower`` holds every
    *fallback_table* value not already covered by ``upper``. A setting left
    to its default in every source stays absent from both, so a ``builtin
    var`` initializer supplies it instead of being suppressed by a host-side
    floor. ``cli_values`` contains only explicitly supplied CLI values; its
    present ``None`` values represent an explicit empty ``Option``.

    ``log`` is never a member of ``upper``/``lower``: it is implied by
    ``log-file`` and derived once per :meth:`EngineSeedTiers.merged` call —
    see that method.

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

    upper: dict[str, Value] = {}
    lower: dict[str, Value] = {}
    for spec in ENGINE_KEYS:
        # ``log`` is implied by log-file, so it follows its own rule in
        # ``EngineSeedTiers`` instead of this per-key resolution.
        if spec.name == "log":
            continue
        if spec.name in cli_values:
            upper[spec.name] = _decode_engine_value(
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

    if "log" in cli_values:
        cli_log: tuple[object, str] | None = (cli_values["log"], "--log")
    elif cli_values.get("log-file") is not None:
        cli_log = (True, "--log-file")
    else:
        cli_log = None

    log_inputs = _LogSeedInputs(
        cli=cli_log,
        upper=_table_log_presence(primary_table),
        lower=_table_log_presence(fallback),
        config_log=config.log,
        config_log_file=config.log_file,
        type_table=type_table,
    )
    return EngineSeedTiers(upper=upper, lower=lower, log_inputs=log_inputs)
