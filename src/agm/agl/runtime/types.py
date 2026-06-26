"""Runtime layer plain service dataclasses.

Host-environment bundle (``HostEnvironment``) and static call/agent/param
declaration summaries (``CallSiteInfo``, ``AgentDeclInfo``, ``ParamDeclInfo``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agm.agl.capabilities import HostCapabilities
    from agm.agl.runtime.agents import AgentRegistry
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.semantics.types import Type as AglType

__all__ = ["AgentDeclInfo", "CallSiteInfo", "HostEnvironment", "ParamDeclInfo"]


@dataclass(frozen=True, slots=True)
class HostEnvironment:
    """Assembled host-runtime environment shared by ``run`` and the REPL session.

    Bundles the three pieces that both the whole-program runner
    (``PipelineDriver.run``) and the incremental ``ReplSession`` need to build
    identically from a set of agent/codec registrations:

    ``registry``
        The ``AgentRegistry`` (named agents + optional default agent).
    ``capabilities``
        The ``HostCapabilities`` static catalog derived from the registry and
        codecs — consumed by the type checker.
    ``codecs``
        The merged ``name → OutputCodec`` table (built-ins + host extras),
        used for contract materialization.
    """

    registry: "AgentRegistry"
    capabilities: "HostCapabilities"
    codecs: dict[str, "OutputCodec"]


@dataclass(frozen=True, slots=True)
class CallSiteInfo:
    """Static summary of one agent-call or exec site (--dry-run inventory).

    ``callee``        Agent or executor name (``"ask"``, ``"exec"``, or a
                      registered agent name).
    ``target_type``   The target type name (e.g. ``"text"``, ``"Review"``).
    ``codec_name``    Selected codec, or ``"none"`` for a ``unit`` target.
    ``has_schema``    ``True`` when the contract carries a JSON Schema.
    ``parse_policy``  ``"abort"`` / ``"retry[N]"`` / ``"default"``.
    ``line``          1-based source line of the call site.
    ``col``           1-based source column of the call site.
    """

    callee: str
    target_type: str
    codec_name: str
    has_schema: bool
    parse_policy: str
    line: int
    col: int


@dataclass(frozen=True, slots=True)
class AgentDeclInfo:
    """Static summary of one ``agent`` declaration in a program.

    ``name``
        The declared agent name.
    ``runner``
        The optional static runner-command hint (a literal string with NO
        interpolation), or ``None`` for a bare ``agent NAME`` declaration.
    ``line``
        1-based source line of the declaration (``span.start_line``).
    ``col``
        1-based source column of the declaration (``span.start_col``).
    """

    name: str
    runner: str | None
    line: int
    col: int


@dataclass(frozen=True, slots=True)
class ParamDeclInfo:
    """Static summary of one ``param`` declaration in a program."""

    name: str
    type: "AglType"
    has_default: bool
    line: int
    col: int
