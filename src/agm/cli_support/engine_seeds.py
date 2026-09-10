"""Shared CLI-over-config resolution of the AgL engine settings a host controls.

``agm exec`` and ``agm repl`` both hand the AgL engine a seed mapping for the
settings the host explicitly controls.  The layering rule is the same for both —
CLI flag, then the command's configuration tables — so it lives here rather than
in either command.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from agm.agent.runner import parse_command
from agm.agl.runtime.engine_config import (
    build_engine_config_seeds,
    convert_config_value,
    raw_option_str,
)
from agm.agl.semantics.engine_keys import get_engine_key_type
from agm.agl.setting_overrides import SettingOverride
from agm.cli_support.agent_values import normalize_agent_source
from agm.config.engine_keys import ENGINE_KEY_NAMES, ENGINE_KEYS, EngineKeyKind, EngineKeySpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agm.agl.semantics.values import Value
    from agm.config.general import ExecConfig

__all__ = ["EngineSeeds", "build_host_engine_seeds", "check_max_iters"]


@dataclass(frozen=True, slots=True)
class EngineSeeds:
    """Host-controlled engine-setting seeds, split by how they reach the program.

    ``values`` are typed ``Value`` seeds threaded straight into the interpreter
    (``PipelineDriver.run_prepared``'s/``ReplSession``'s ``builtin_host_settings``
    / ``engine_base``), exactly as every non-agent engine key already works.

    ``overrides`` are host-supplied AgL source text for ``std/config`` keys —
    currently only a normalized ``default-agent`` from ``--agent`` or
    ``[exec] default-agent`` — meant for
    ``PipelineDriver.prepare_parsed_entry``'s (or the REPL's) ``setting_overrides``
    seam, so the literal is resolved, type-checked, and constant-checked by the
    program's own compilation rather than a separate throwaway one.

    A key never appears in both mappings.
    """

    values: "dict[str, Value]" = field(default_factory=dict)
    overrides: "dict[str, SettingOverride]" = field(default_factory=dict)


def check_max_iters(max_iters: int | None) -> None:
    """Reject a non-positive ``--max-iters`` before anything runs.

    The ``max-iters`` safety valve counts iterations, so zero and negatives are
    meaningless; ``None`` means the flag was not given.  Shared by ``agm exec``
    and ``agm repl``, which take the flag with identical semantics.
    """
    if max_iters is not None and max_iters <= 0:
        print("Error: --max-iters must be a positive integer", file=sys.stderr)
        raise SystemExit(1)


def _require_agent_text(value: object, *, source: str) -> str:
    """Validate and normalize one host-facing Agent value."""
    if not isinstance(value, str) or not value.strip():
        print(
            f"Error: invalid default-agent value from {source}: "
            f"expected a non-empty Agent value, got {value!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return normalize_agent_source(value)


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


def build_host_engine_seeds(
    *,
    config: "ExecConfig",
    primary_table: "Mapping[str, object]",
    fallback_table: "Mapping[str, object] | None" = None,
    cli_values: "Mapping[str, object | None]",
    agent: str | None,
) -> EngineSeeds:
    """Decode the engine settings the host explicitly controls into seed values.

    Only explicit controls are seeded: a setting left to its default stays
    absent from the result, so a ``builtin var`` initializer supplies it instead
    of being suppressed by a host-side floor. ``cli_values`` contains only
    explicitly supplied CLI values; its present ``None`` values represent an
    explicit empty ``Option``. CLI values win over configuration tables, which
    are consulted in *primary_table* then *fallback_table* order.

    ``default-agent`` precedence, highest first: ``--agent``, then
    the qualified program table/``[exec] default-agent``, then ``[exec] runner``.  Exactly
    one of the three ever supplies the key, and it lands in exactly one of the
    two result mappings: an Agent value (``--agent``/``default-agent``) is
    normalized into constructor source and becomes a
    :class:`~agm.agl.setting_overrides.SettingOverride` in ``overrides`` so the
    program's own compilation resolves it; a bare host command
    (``[exec] runner``) is decoded directly into an ``AgentCommand`` value in
    ``values``, since it is host text, not AgL source — rendering it as AgL
    source would have to re-escape quotes, backslashes, and ``%{``, which
    ``AgentCommand`` deliberately carries verbatim.  A blank or non-string raw
    config/CLI value exits 1 here, before anything runs.  ``[exec] runner`` is
    additionally shell-split with :func:`~agm.agent.runner.parse_command` right
    here, so a malformed command (e.g. an unclosed quote) also exits 1 before
    anything runs rather than surfacing later as a runtime agent-call failure.

    The two normalized Agent sources carry different
    :attr:`~agm.agl.setting_overrides.SettingOverride.required` provenance:
    ``--agent`` is an explicit per-run request, so it is marked
    ``required=True`` and still produces a diagnostic when the loaded program
    never brings in ``std/config`` (e.g. ``--no-stdlib``); ``[exec]``/
    qualified program-table ``default-agent`` is ambient configuration, marked
    ``required=False``, so it is simply inert — no diagnostic — in that same
    situation.
    """
    fallback: "Mapping[str, object]" = fallback_table if fallback_table is not None else {}
    configured = {key for key in ENGINE_KEY_NAMES if key in primary_table or key in fallback}

    seed_raw: dict[str, object] = {}
    for spec in ENGINE_KEYS:
        # ``log`` is implied by log-file and ``default-agent`` has distinct
        # source-text/runner precedence, so neither follows per-key resolution.
        if spec.name in {"log", "default-agent"}:
            continue
        if spec.name in cli_values:
            seed_raw[spec.name] = cli_values[spec.name]
        elif spec.name in configured:
            value = _configured_value(spec, config, primary_table, fallback)
            # A ``None`` config result is absent, not an explicit control. In
            # particular this lets a builtin initializer supply invalid/empty
            # max-iters and Option values just as before.
            if value is not None:
                seed_raw[spec.name] = value

    # ``log`` is not an ordinary setting: a supplied log-file implies it. Keep
    # that relationship explicit instead of encoding it in the catalog loop.
    if "log" in cli_values:
        seed_raw["log"] = cli_values["log"]
    elif cli_values.get("log-file") is not None:
        seed_raw["log"] = True
    elif configured & {"log", "log-file"}:
        seed_raw["log"] = config.log or config.log_file is not None

    seeds = build_engine_config_seeds(seed_raw)
    overrides: dict[str, SettingOverride] = {}

    if agent is not None:
        literal = _require_agent_text(agent, source="--agent")
        overrides["default-agent"] = SettingOverride(
            source=literal, origin="--agent", required=True
        )
    elif config.default_agent is not None:
        literal = _require_agent_text(config.default_agent, source="[exec] configuration")
        overrides["default-agent"] = SettingOverride(
            source=literal, origin="[exec] default-agent", required=False
        )
    elif config.runner is not None:
        try:
            parse_command(config.runner, kind="[exec] runner")
        except ValueError as exc:
            print(f"Error: {exc}.", file=sys.stderr)
            raise SystemExit(1) from exc
        agent_type = get_engine_key_type("default-agent")
        assert agent_type is not None
        seeds["default-agent"] = convert_config_value(
            "default-agent",
            {"$case": "AgentCommand", "command": config.runner},
            agent_type,
        )

    return EngineSeeds(values=seeds, overrides=overrides)
