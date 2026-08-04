"""Default runner for ``agm loop``.

A pure data leaf: it imports nothing, so loop code can depend on it without
pulling in config loading or process execution.
"""

from __future__ import annotations

# The runner command used when loop CLI flags and config leave it unset.
# ``-p`` runs the agent non-interactively, which every caller requires: an
# interactive runner would block waiting on a terminal that is not there.
DEFAULT_AGENT_RUNNER = "claude -p"
