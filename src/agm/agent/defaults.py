"""Built-in runner floor shared by ``agm loop``, ``agm review``, and ``agm revise``.

A pure data leaf: it imports nothing, so those commands can depend on it
without pulling in config loading or process execution. Each command reads its
own CLI arguments and its own config section, so a shared floor keeps them
independent of one another.
"""

from __future__ import annotations

# The runner command used when a command's CLI flags and config leave it
# unset. ``-p`` runs the agent non-interactively, which every caller requires:
# an interactive runner would block waiting on a terminal that is not there.
DEFAULT_AGENT_RUNNER = "claude -p"
