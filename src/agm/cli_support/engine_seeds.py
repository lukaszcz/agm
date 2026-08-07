"""Shared CLI-over-config resolution of the AgL engine settings a host controls.

``agm exec`` and ``agm repl`` both hand the AgL engine a seed mapping for the
settings the host explicitly controls.  The layering rule is the same for both —
CLI flag, then the command's configuration tables — so it lives here rather than
in either command.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agm.agent.runner import parse_command
from agm.agl.runtime.params import build_engine_config_seeds, convert_config_value, raw_option_str
from agm.agl.semantics.engine_keys import get_engine_key_type
from agm.agl.setting_overrides import SettingOverride
from agm.config.engine_keys import ENGINE_KEY_NAMES

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
    currently only ``default-agent`` when it comes from an AgL literal
    (``--agent`` or ``[exec] default-agent``) — meant for
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


def _require_agent_literal_text(literal: object, *, source: str) -> str:
    """Validate that a raw host value is usable as AgL ``Agent`` literal source.

    This is the one check that must happen before the literal is handed to the
    AgL pipeline at all: a non-string or blank value is a host-shape error, not
    an AgL question, so it is diagnosed here rather than surfacing as a
    confusing parse failure downstream.
    """
    if not isinstance(literal, str) or not literal.strip():
        print(
            f"Error: invalid default-agent literal from {source}: "
            f"expected a non-empty AgL Agent literal, got {literal!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return literal


def build_host_engine_seeds(
    *,
    config: "ExecConfig",
    primary_table: "Mapping[str, object]",
    fallback_table: "Mapping[str, object] | None" = None,
    log_enabled: bool,
    strict_json: bool | None,
    max_iters: int | None,
    log: bool,
    no_log: bool,
    log_file: str | None,
    agent: str | None,
    timeout: str | None = None,
    no_timeout: bool = False,
    no_log_file: bool = False,
) -> EngineSeeds:
    """Decode the engine settings the host explicitly controls into seed values.

    Only explicit controls are seeded: a setting left to its default stays
    absent from the result, so a ``builtin var`` initializer supplies it instead
    of being suppressed by a host-side floor.  CLI flags win over the
    configuration tables, which are consulted in *primary_table* then
    *fallback_table* order.  ``--no-timeout``/``--no-log-file`` seed an explicit
    empty ``Option``, which is a control rather than an absence; commands
    without those flags leave them unset.

    ``default-agent`` precedence, highest first: ``--agent``, then
    ``[exec]``/``[<program>] default-agent``, then ``[exec] runner``.  Exactly
    one of the three ever supplies the key, and it lands in exactly one of the
    two result mappings: an AgL literal (``--agent``/``default-agent``) becomes
    a :class:`~agm.agl.setting_overrides.SettingOverride` in ``overrides`` so
    the program's own compilation resolves it; a bare host command
    (``[exec] runner``) is decoded directly into an ``AgentCommand`` value in
    ``values``, since it is host text, not AgL source — rendering it as AgL
    source would have to re-escape quotes, backslashes, and ``%{``, which
    ``AgentCommand`` deliberately carries verbatim.  A blank or non-string raw
    config/CLI value exits 1 here, before anything runs.  ``[exec] runner`` is
    additionally shell-split with :func:`~agm.agent.runner.parse_command` right
    here, so a malformed command (e.g. an unclosed quote) also exits 1 before
    anything runs rather than surfacing later as a runtime agent-call failure.
    """
    fallback: "Mapping[str, object]" = fallback_table if fallback_table is not None else {}
    configured = {key for key in ENGINE_KEY_NAMES if key in primary_table or key in fallback}

    seed_raw: dict[str, object] = {}
    if strict_json is not None:
        seed_raw["strict-json"] = strict_json
    elif "strict-json" in configured:
        seed_raw["strict-json"] = config.strict_json

    if max_iters is not None:
        seed_raw["max-iters"] = max_iters
    elif "max-iters" in configured and config.default_loop_limit is not None:
        seed_raw["max-iters"] = config.default_loop_limit

    if timeout is not None:
        seed_raw["timeout"] = timeout
    elif no_timeout:
        seed_raw["timeout"] = None
    elif "timeout" in configured:
        # The raw spelling (e.g. "30s") is preserved for the Option[text] key.
        raw_timeout = raw_option_str(primary_table, fallback, "timeout")
        if raw_timeout is not None:
            seed_raw["timeout"] = raw_timeout

    if no_log or log or log_file is not None or configured & {"log", "log-file"}:
        seed_raw["log"] = log_enabled

    if log_file is not None:
        seed_raw["log-file"] = log_file
    elif no_log_file:
        seed_raw["log-file"] = None
    elif "log-file" in configured and config.log_file is not None:
        seed_raw["log-file"] = config.log_file

    seeds = build_engine_config_seeds(seed_raw)
    overrides: dict[str, SettingOverride] = {}

    if agent is not None:
        literal = _require_agent_literal_text(agent, source="--agent")
        overrides["default-agent"] = SettingOverride(source=literal, origin="--agent")
    elif config.default_agent is not None:
        literal = _require_agent_literal_text(config.default_agent, source="[exec] configuration")
        overrides["default-agent"] = SettingOverride(source=literal, origin="[exec] default-agent")
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
