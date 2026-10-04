"""Call-scoped agent services for standard-library companions."""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agm.agl.ir.builtin_nominals import BuiltinNominals
from agm.util.scoping import ScopedVar

if TYPE_CHECKING:
    from agm.agent.spec_defaults import AgentSpecResolver
    from agm.sandbox.prepare import SandboxContext


@dataclass(frozen=True, slots=True)
class HostAgentServices:
    """Agent identities, configured defaults, and sandbox preparation owned by the host."""

    nominals: BuiltinNominals
    resolve_agent_spec: AgentSpecResolver | None
    get_sandbox_context: Callable[[], SandboxContext] | None


_ACTIVE_SERVICES: contextvars.ContextVar[HostAgentServices] = contextvars.ContextVar(
    "agl_host_agent_services"
)


def active_agent_services(services: HostAgentServices) -> ScopedVar[HostAgentServices]:
    """Publish the host's services for an extern call's extent."""
    return ScopedVar(_ACTIVE_SERVICES, services)


def current_agent_services() -> HostAgentServices:
    """Return the agent services of the current extern call."""
    return _ACTIVE_SERVICES.get()
