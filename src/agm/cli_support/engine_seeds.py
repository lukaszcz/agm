"""Shared CLI-over-config resolution of the AgL engine settings a host controls.

``agm exec`` and ``agm repl`` both hand the AgL engine a seed mapping for the
settings the host explicitly controls.  The layering rule is the same for both —
CLI flag, then the command's configuration tables — so it lives here rather than
in either command.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from agm.agl.runtime.params import build_engine_config_seeds, raw_option_str
from agm.agl.semantics.engine_keys import get_engine_key_type
from agm.config.engine_keys import ENGINE_KEY_NAMES

if TYPE_CHECKING:
    from collections.abc import Mapping

    from agm.agl.semantics.values import Value
    from agm.config.general import ExecConfig

__all__ = ["build_host_engine_seeds", "check_max_iters", "parse_default_agent_literal"]


def check_max_iters(max_iters: int | None) -> None:
    """Reject a non-positive ``--max-iters`` before anything runs.

    The ``max-iters`` safety valve counts iterations, so zero and negatives are
    meaningless; ``None`` means the flag was not given.  Shared by ``agm exec``
    and ``agm repl``, which take the flag with identical semantics.
    """
    if max_iters is not None and max_iters <= 0:
        print("Error: --max-iters must be a positive integer", file=sys.stderr)
        raise SystemExit(1)


def parse_default_agent_literal(literal: object, *, source: str) -> "Value":
    """Parse a host-supplied ``Agent`` literal into a typed AgL value."""
    if not isinstance(literal, str) or not literal.strip():
        raise ValueError(
            f"invalid default-agent literal from {source}: "
            f"expected a non-empty AgL Agent literal, got {literal!r}"
        )

    from agm.agl.constant import ConstantExpressionError, parse_constant

    expected_type = get_engine_key_type("default-agent")
    assert expected_type is not None
    try:
        return parse_constant(literal, expected_type)
    except ConstantExpressionError as exc:
        raise ValueError(f"invalid default-agent literal from {source}: {exc}") from exc


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
) -> "dict[str, Value]":
    """Decode the engine settings the host explicitly controls into seed values.

    Only explicit controls are seeded: a setting left to its default stays
    absent from the result, so a ``builtin var`` initializer supplies it instead
    of being suppressed by a host-side floor.  CLI flags win over the
    configuration tables, which are consulted in *primary_table* then
    *fallback_table* order.  ``--no-timeout``/``--no-log-file`` seed an explicit
    empty ``Option``, which is a control rather than an absence; commands
    without those flags leave them unset.

    An unparseable ``Agent`` literal exits 1 here, before anything runs.
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

    default_agent_literal = agent if agent is not None else config.default_agent
    if default_agent_literal is not None:
        source = "--agent" if agent is not None else "[exec] configuration"
        try:
            seeds["default-agent"] = parse_default_agent_literal(
                default_agent_literal, source=source
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    return seeds
