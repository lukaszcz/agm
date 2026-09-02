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
    "ENTRY_PARAM_QUALIFIER",
    "CallSiteInfo",
    "HostEnvironment",
    "ParamDeclInfo",
    "ProgramDeclInfo",
    "ProgramParamInfo",
    "public_param_spelling",
]

ENTRY_PARAM_QUALIFIER = "@entry"
"""The shell-safe namespace a param of an unnamed entry module is addressed in."""


def public_param_spelling(qualified_name: str, *, entry_qualifier: str | None = None) -> str:
    """Return the user-facing spelling of a module-qualified param name.

    The entry module is an internal identity with no user-facing name, so a
    qualified spelling that leaks its sentinel is meaningless in a diagnostic
    or a CLI flag. A file-backed entry is addressed by its module route
    (*entry_qualifier*); every other entry by the reserved
    :data:`ENTRY_PARAM_QUALIFIER` namespace, which no module route can claim.
    Names qualified by a real module are returned unchanged.
    """
    qualifier, separator, remainder = qualified_name.partition("::")
    if not separator or qualifier != ENTRY_DISPLAY:
        return qualified_name
    return f"{entry_qualifier or ENTRY_PARAM_QUALIFIER}{separator}{remainder}"


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
    ``parameters`` is the program's own value-parameter signature, in
    declaration order.
    """

    module: "ModuleId"
    scope_path: tuple[str, ...]
    name: str
    node_id: int
    parameters: tuple[ProgramParamInfo, ...] = ()

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

    ``module_segments`` and ``name`` together retain the declaration's
    external identity. ``name`` is its scope-path spelling: a root param's
    bare name, or a scoped param's full ``::``-joined path spelling. The
    module-qualified spelling disambiguates same-named params in one program
    inventory. ``is_entry`` keeps the synthetic entry-module identity separate
    from its user-facing option spelling. ``entry_qualifier`` supplies that
    spelling for file-backed entries without changing their internal identity.
    """

    name: str
    type: "AglType"
    has_default: bool
    line: int
    col: int
    module_segments: tuple[str, ...] = ()
    is_entry: bool = False
    entry_qualifier: str | None = None

    @property
    def qualified_name(self) -> str:
        """Return the module-qualified external spelling."""
        if not self.module_segments:
            return self.name
        return f"{'/'.join(self.module_segments)}::{self.name}"
