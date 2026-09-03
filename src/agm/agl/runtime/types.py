"""Runtime layer plain service dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agm.agl.modules.ids import ENTRY_DISPLAY

if TYPE_CHECKING:
    from agm.agl.capabilities import HostCapabilities
    from agm.agl.ir.zones import ParamZone
    from agm.agl.modules.ids import ModuleId
    from agm.agl.runtime.agents import AgentFn
    from agm.agl.runtime.codec import OutputCodec
    from agm.agl.runtime.externs import ExternRegistry
    from agm.agl.runtime.sessions import SessionHost
    from agm.agl.semantics.types import Type as AglType
    from agm.agl.syntax.spans import SourceSpan

__all__ = [
    "CallSiteInfo",
    "HostEnvironment",
    "ProgramDeclInfo",
    "ProgramParamInfo",
    "public_param_spelling",
]


def public_param_spelling(qualified_name: str) -> str:
    """Return the user-facing spelling of a module-qualified param name.

    The entry module is an internal identity with no user-facing name, so a
    qualified spelling that leaks its sentinel is meaningless in a diagnostic.
    When *qualified_name* is qualified by the entry sentinel, the sentinel is
    stripped and the bare remainder — the name as the author wrote it — is
    returned. A name qualified by a real module is returned unchanged, since
    that qualification is what distinguishes two same-named params.
    """
    qualifier, separator, remainder = qualified_name.partition("::")
    if not separator or qualifier != ENTRY_DISPLAY:
        return qualified_name
    return remainder


@dataclass(frozen=True, slots=True)
class HostEnvironment:
    """Assembled host-runtime environment shared by ``run`` and the REPL session.

    Bundles the services that both the whole-program runner
    (``PipelineDriver.run``) and the incremental ``ReplSession`` use:

    ``agent_dispatcher``
        The value-driven host dispatcher for ``Agent`` enum values.
    ``session_host``
        The opaque lifecycle service used by persistent ``Session`` values.
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
    session_host: "SessionHost | None"
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
class ProgramParamInfo:
    """Static summary of one ``program def`` value parameter, as the host sees it.

    ``name``         — the declared parameter name.
    ``kind``         — the parameter's zone (positional-only, standard,
                        named-only), governing how a host projects it.
    ``type``         — the parameter's checked type.
    ``has_default``  — ``True`` when the parameter has a default expression.
    ``span``         — the parameter's declaration span, the anchor for a
                        binding or decode diagnostic naming this parameter.
    """

    name: str
    kind: "ParamZone"
    type: "AglType"
    has_default: bool
    span: "SourceSpan"


@dataclass(frozen=True, slots=True)
class ProgramDeclInfo:
    """Static summary of one ``program def`` declaration.

    ``module`` and ``scope_path`` retain the declaration identity in structured
    form. ``declaration_path`` is the external spelling within that module;
    ``qualified_path`` prefixes it with a non-entry module route.
    ``span`` is the declaration's own span, the anchor for a diagnostic that
    names no single parameter (an unknown argument name, or an excess
    positional argument). ``parameters`` is the program's own value-parameter
    signature, in declaration order.
    """

    module: "ModuleId"
    scope_path: tuple[str, ...]
    name: str
    node_id: int
    span: "SourceSpan"
    parameters: tuple[ProgramParamInfo, ...]

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
