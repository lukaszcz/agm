"""AgentRegistry for the AgL runtime.

``AgentRegistry`` holds the callable agents registered with a
``WorkflowRuntime``.  It distinguishes:

- **Named agents**: registered with ``register_agent(name, fn)``.
- **Default agent** (``prompt``): registered via the ``default_agent``
  constructor kwarg; handles the built-in ``prompt`` contextual keyword.
- **Fallback support**: when ``has_fallback`` is ``True``, any unrecognised
  agent name is dispatched to the default agent (used by the CLI runtime that
  wires the runner-backed default agent as a fallback for all names).

The registry is also the source of the ``HostCapabilities.has_fallback_agent``
and ``HostCapabilities.agent_names`` values.
"""

from __future__ import annotations

from collections.abc import Callable

from agm.agl.runtime.request import AgentRequest, AgentResponse

# A host agent callable may return a plain ``str`` or a full ``AgentResponse``.
AgentFn = Callable[[AgentRequest], AgentResponse | str]


class AgentRegistry:
    """Immutable-after-build registry of host agents.

    Parameters
    ----------
    named:
        Pre-validated mapping from agent name → callable.
    default_agent:
        The callable for the built-in ``prompt`` keyword (or ``None`` if
        no default agent is configured).
    """

    def __init__(
        self,
        *,
        named: dict[str, AgentFn],
        default_agent: AgentFn | None,
    ) -> None:
        self._named = named
        self._default = default_agent

    @property
    def has_fallback(self) -> bool:
        """True when a default agent is configured.

        With a default agent, any unknown agent name falls back to it, so
        the capability checker treats all names as valid.
        """
        return self._default is not None

    @property
    def agent_names(self) -> frozenset[str]:
        """Names of explicitly registered named agents."""
        return frozenset(self._named)

    def dispatch(self, name: str, request: AgentRequest) -> AgentResponse:
        """Dispatch a call to the appropriate agent callable.

        Resolution order:
        1. Named agents (exact match).
        2. Default agent fallback (when ``has_fallback`` is True).

        Raises ``KeyError`` if the name is unknown and no fallback exists.
        """
        fn: AgentFn | None = self._named.get(name)
        if fn is None:
            if name == "prompt" and self._default is not None:
                fn = self._default
            elif self._default is not None:
                # Named agent not registered → fall back to default agent.
                fn = self._default
            else:
                raise KeyError(f"No agent registered for name {name!r}")
        raw = fn(request)
        if isinstance(raw, str):
            return AgentResponse(content=raw)
        return raw
