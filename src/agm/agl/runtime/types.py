"""Runtime layer plain service dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agm.agl.capabilities import HostCapabilities
    from agm.agl.modules.ids import ModuleId
    from agm.agl.runtime.agents import AgentFn
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.externs import ExternRegistry
    from agm.agl.semantics.types import Type as AglType

__all__ = [
    "CallSiteInfo",
    "HostEnvironment",
    "ParamDeclInfo",
    "ProgramDeclInfo",
]


@dataclass(frozen=True, slots=True)
class HostEnvironment:
    """Assembled host-runtime environment shared by ``run`` and the REPL session.

    Bundles the services that both the whole-program runner
    (``PipelineDriver.run``) and the incremental ``ReplSession`` use:

    ``agent_dispatcher``
        The value-driven host dispatcher for ``Agent`` enum values.
    ``capabilities``
        The ``HostCapabilities`` static catalog derived from codecs — consumed
        by the type checker.
    ``codecs``
        The merged ``name → OutputCodec`` table (built-ins + host extras),
        used for contract materialization.
    ``extern_registry``
        The ``ExternRegistry`` used to import extern-def companion modules
        and resolve their callables.  Built empty; the pipeline populates it
        from a program's loaded modules before evaluation.
    """

    agent_dispatcher: "AgentFn | None"
    capabilities: "HostCapabilities"
    codecs: dict[str, "OutputCodec"]
    extern_registry: "ExternRegistry"


@dataclass(frozen=True, slots=True)
class CallSiteInfo:
    """Static summary of one agent-call or exec site (--dry-run inventory).

    ``callee``        Agent or executor name (``"ask"`` or ``"exec"``).
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
class ProgramDeclInfo:
    """Static summary of one ``program def`` declaration.

    ``module`` and ``scope_path`` retain the declaration identity in structured
    form. ``declaration_path`` is the external spelling within that module;
    ``qualified_path`` prefixes it with a non-entry module route.
    """

    module: "ModuleId"
    scope_path: tuple[str, ...]
    name: str
    node_id: int

    @property
    def declaration_path(self) -> str:
        """Return the program's scope-qualified declaration spelling."""
        return "::".join((*self.scope_path, self.name))

    @property
    def qualified_path(self) -> str:
        """Return the program spelling qualified by its module when available."""
        if self.module.is_entry:
            return self.declaration_path
        return f"{self.module.path_str()}::{self.declaration_path}"


@dataclass(frozen=True, slots=True)
class ParamDeclInfo:
    """Static summary of one ``param`` declaration in a program.

    ``name`` is the param's external key: a root param's bare name, or a
    scoped param's full ``::``-joined path spelling (e.g. ``"Deploy::region"``)
    — the same string the CLI flag and config-table key use.
    """

    name: str
    type: "AglType"
    has_default: bool
    line: int
    col: int
