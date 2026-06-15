"""Shared, mutable agent-call mode holder for the AgL REPL.

The REPL can dispatch live agent (and ``exec`` shell) calls in one of two modes:

- ``"confirm"`` — confirm before every live call (the eventual default per the
  REPL plan, decision 2);
- ``"auto"``    — fire calls immediately, like ``agm exec``.

This tiny module exists on its own so BOTH the meta-command layer
(:mod:`agm.agl.repl.meta`, which mutates the mode via ``:agent confirm|auto``)
and M4's confirming agent wrapper (which will *read* the mode before each call)
can import it without a cyclic dependency.

**M4 wiring seam.** ``AgentMode`` is a mutable holder, not a value: the meta
layer mutates ``mode`` in place. M4 must construct ONE :class:`AgentMode` and
pass the SAME instance to both :func:`agm.agl.repl.console.run_console` (so
``:agent`` mutates it) and to the confirming wrapper (so the wrapper sees the
mutation). Until M4 wires that wrapper, ``:agent`` changes this holder but has
**no observable effect** on evaluation — the mode is recorded and reported, and
nothing reads it yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AgentModeName = Literal["confirm", "auto"]


@dataclass(slots=True)
class AgentMode:
    """Mutable holder for the current agent-call confirmation mode.

    ``mode`` defaults to ``"confirm"`` per plan decision 2. The meta layer
    mutates this field; M4's confirming wrapper will read it.
    """

    mode: AgentModeName = "confirm"
