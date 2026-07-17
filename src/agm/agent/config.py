"""Agent configuration helpers."""

from __future__ import annotations

from agm.agent.defaults import DEFAULT_AGENT_RUNNER
from agm.config.context import current_config_context
from agm.config.general import load_loop_config, loop_config_from_merged
from agm.core.toml import TomlDict


def default_agent_runner(*, merged: TomlDict | None = None) -> str:
    """Resolve the shared default agent runner from the ``[loop]`` config.

    When *merged* is supplied, the ``[loop]`` section is derived from it,
    avoiding a redundant config-file load for callers that already hold a
    merged config.
    """
    if merged is not None:
        loop_config = loop_config_from_merged(merged)
    else:
        context = current_config_context()
        loop_config = load_loop_config(
            home=context.home,
            proj_dir=context.proj_dir,
            cwd=context.cwd,
        )
    return loop_config.runner if loop_config.runner is not None else DEFAULT_AGENT_RUNNER
