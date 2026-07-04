"""Shared, mutable agent-call mode holder for the AgL REPL.

The REPL can dispatch live agent (and ``exec`` shell) calls in one of two modes:

- ``"confirm"`` — confirm before every live call;
- ``"auto"``    — fire calls immediately, like ``agm exec``.

This tiny module exists on its own so BOTH the meta-command layer
(:mod:`agm.agl.repl.meta`, which mutates the mode via ``:agent confirm|auto``)
and the confirming agent wrapper (:class:`agm.agl.repl.agents.ConfirmingAgent`,
which *reads* the mode before each live call) can import it without a cyclic
dependency.

``AgentMode`` is a mutable holder, not a value: the meta layer mutates ``mode``
in place. The REPL constructs ONE :class:`AgentMode` and passes the SAME
instance to both the console (so ``:agent`` mutates it) and the confirming
wrapper (so the wrapper sees the mutation on the next call).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AgentModeName = Literal["confirm", "auto"]


@dataclass(slots=True)
class AgentMode:
    """Mutable holder for the current agent-call confirmation mode.

    ``mode`` defaults to ``"confirm"``. The meta layer mutates this field; the
    confirming wrapper reads it before each call.
    """

    mode: AgentModeName = "confirm"
