"""Agent defaults shared across the CLI, the loop workflows, and the AgL runtime.

A pure data leaf: it imports nothing, so any layer can depend on it without
pulling in config loading or process execution.
"""

from __future__ import annotations

# The runner command used when neither a CLI flag nor config selects one.
# ``-p`` runs the agent non-interactively, which every caller requires: an
# interactive runner would block waiting on a terminal that is not there.
DEFAULT_AGENT_RUNNER = "claude -p"
