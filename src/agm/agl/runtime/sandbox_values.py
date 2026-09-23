"""Value-driven AgL ``AgentSandbox``/``Sandbox`` decoding, beside ``agents.py``.

The value<->host boundary for the ``sandbox`` named parameter threaded
through ``ask``/``ask-request``/``Session::open`` and their receiver forms:
decodes the AgL ``AgentSandbox`` value a call site supplies into
:class:`AgentSandboxMode`, the canonical decoded shape, and encodes it back.
Member identity is resolved through ``resolve_standard_member_name``
(nominal identity), never by reading a name off the value itself, exactly as
``agents.py`` resolves an ``Agent`` member.

``AgentSandboxMode`` is a closed union mirroring ``AgentSandbox``'s three
members: ``Sandboxed`` is the only case that carries data (its
``SandboxLimits``, always present), so a caller can never construct "native
plus limits" or "sandboxed without limits" -- the shape itself rules it out,
rather than a convention downstream code has to honor.
:func:`permission_mode_and_limits` is the one place this union is converted
to the host's ``(PermissionMode, SandboxLimits | None)`` pair that process
preparation understands; every such call site reuses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agm.agent.spec import PermissionMode
from agm.agl.ir.builtin_nominals import BuiltinNominals, resolve_standard_member_name
from agm.agl.ir.reserved_nominals import AGENT_SANDBOX_MEMBERS
from agm.agl.runtime.option import none_value, option_text, some_value
from agm.agl.semantics.values import BoolValue, RecordValue, TextValue, Value
from agm.sandbox.request import Default, DefaultLimit, LimitSpec, SandboxLimits

__all__ = [
    "AgentSandboxMode",
    "Disabled",
    "Native",
    "Sandboxed",
    "agent_sandbox_value",
    "decode_agent_sandbox",
    "decode_sandbox_record",
    "permission_mode_and_limits",
    "sandbox_limits_value",
    "sandbox_mode_from_permission",
]

_OPTIONAL_MEMBERS = ("Some", "None", "Default")


@dataclass(frozen=True, slots=True)
class Disabled:
    """``AgentSandbox::Disabled``: no sandboxing, agent runs with its own default permissions."""


@dataclass(frozen=True, slots=True)
class Native:
    """``AgentSandbox::Native``: the agent's own native (unsandboxed) permission mode."""


@dataclass(frozen=True, slots=True)
class Sandboxed:
    """``AgentSandbox::Sandbox(...)``: runs under the sandbox with these limits."""

    limits: SandboxLimits


#: Closed union mirroring the AgL ``AgentSandbox`` enum: the canonical
#: decoded shape every ``sandbox`` operand and stored session settles to.
AgentSandboxMode = Disabled | Native | Sandboxed


def decode_agent_sandbox(value: RecordValue, nominals: BuiltinNominals) -> AgentSandboxMode:
    """Decode an ``AgentSandbox`` member value into its canonical union shape.

    ``Disabled`` and ``Native`` carry no data; the referenced ``Sandbox``
    member decodes its own record through :func:`decode_sandbox_record`.
    """
    case = resolve_standard_member_name(
        value.nominal, "AgentSandbox", AGENT_SANDBOX_MEMBERS, nominals
    )
    if case == "Disabled":
        return Disabled()
    if case == "Native":
        return Native()
    if case == "Sandbox":
        return Sandboxed(decode_sandbox_record(value, nominals))
    raise ValueError("value is not a recognized AgentSandbox member")


def decode_sandbox_record(value: RecordValue, nominals: BuiltinNominals) -> SandboxLimits:
    """Decode a ``Sandbox`` record value into the host ``SandboxLimits`` it names.

    ``profile_name`` is not decided here: the caller supplies it once it
    knows the command (the agent's executable for a dispatch, ``exec``'s
    first shell word).
    """
    return SandboxLimits(
        memory=_decode_limit(value.fields["memory"], nominals),
        swap=_decode_limit(value.fields["swap"], nominals),
        settings_file=_decode_settings_file(value.fields["settings"], nominals),
        patch=_decode_bool(value.fields["patch"]),
    )


def permission_mode_and_limits(
    mode: AgentSandboxMode,
) -> tuple[PermissionMode, SandboxLimits | None]:
    """Convert a decoded ``AgentSandboxMode`` to the host's dispatch pair.

    The one place this union becomes the ``(PermissionMode, SandboxLimits |
    None)`` pair every process-preparation call site (agent dispatch, session
    open) consumes; reuse this rather than matching the union again at a
    second call site.
    """
    if isinstance(mode, Native):
        return PermissionMode.NATIVE, None
    if isinstance(mode, Sandboxed):
        return PermissionMode.UNRESTRICTED, mode.limits
    return PermissionMode.NONE, None


def sandbox_mode_from_permission(
    permission_mode: PermissionMode, limits: SandboxLimits | None
) -> AgentSandboxMode:
    """Rebuild the decoded union from a host pair produced by :func:`permission_mode_and_limits`.

    Used only to read a session's fixed-at-open mode back from host-stored
    state (a session snapshot), never to decide sandboxing. Total over
    ``PermissionMode``'s three members: ``UNRESTRICTED`` always carries the
    limits ``permission_mode_and_limits`` gave it, so a missing pairing here
    means the pair did not come from this module and is rejected rather than
    silently patched with a fabricated default.
    """
    if permission_mode is PermissionMode.NATIVE:
        return Native()
    if permission_mode is PermissionMode.UNRESTRICTED:
        if limits is None:
            raise ValueError("UNRESTRICTED permission mode requires sandbox limits")
        return Sandboxed(limits)
    return Disabled()


def agent_sandbox_value(mode: AgentSandboxMode, nominals: BuiltinNominals) -> RecordValue:
    """Encode a decoded ``AgentSandboxMode`` into its AgL ``AgentSandbox`` value.

    ``Sandboxed`` always carries the ``SandboxLimits`` its member value
    needs, so this can never produce a member silently missing the limits it
    was handed.
    """
    if isinstance(mode, Native):
        return RecordValue(
            nominal=nominals.resolve_standard_member("AgentSandbox", "Native").nominal, fields={}
        )
    if isinstance(mode, Sandboxed):
        return sandbox_limits_value(mode.limits, nominals)
    return RecordValue(
        nominal=nominals.resolve_standard_member("AgentSandbox", "Disabled").nominal, fields={}
    )


def sandbox_limits_value(limits: SandboxLimits, nominals: BuiltinNominals) -> RecordValue:
    """Encode host ``SandboxLimits`` into its AgL ``Sandbox`` record value.

    The ``Sandbox`` record's identity IS ``AgentSandbox::Sandbox``'s
    (enum-record subsumption shares one nominal id), so this value is valid
    wherever either type is expected.
    """
    return RecordValue(
        nominal=nominals.resolve("Sandbox").nominal,
        fields={
            "memory": _limit_value(limits.memory, nominals),
            "swap": _limit_value(limits.swap, nominals),
            "settings": _settings_value(limits.settings_file, nominals),
            "patch": BoolValue(limits.patch),
        },
    )


def _decode_limit(value: Value, nominals: BuiltinNominals) -> LimitSpec:
    """Decode an ``Optional[text]`` limit field: ``Default``/``None``/``Some(text)``."""
    assert isinstance(value, RecordValue)
    case = resolve_standard_member_name(value.nominal, "Optional", _OPTIONAL_MEMBERS, nominals)
    if case == "Default":
        return Default
    if case == "None":
        return None
    if case == "Some":
        payload = value.fields["value"]
        assert isinstance(payload, TextValue)
        return payload.value
    raise ValueError("value is not a recognized Optional[text] member")


def _limit_value(limit: LimitSpec, nominals: BuiltinNominals) -> RecordValue:
    """Encode a ``LimitSpec`` back into its ``Optional[text]`` member value."""
    if isinstance(limit, DefaultLimit):
        return RecordValue(
            nominal=nominals.resolve_standard_member("Optional", "Default").nominal, fields={}
        )
    if limit is None:
        return RecordValue(
            nominal=nominals.resolve_standard_member("Optional", "None").nominal, fields={}
        )
    return RecordValue(
        nominal=nominals.resolve_standard_member("Optional", "Some").nominal,
        fields={"value": TextValue(limit)},
    )


def _decode_settings_file(value: Value, nominals: BuiltinNominals) -> Path | None:
    """Decode an ``Option[text]`` settings-path field.

    ``Path`` normalizes the text (trailing separators, ``./`` segments), so a
    value later encoded from the result may differ textually from what was
    written here while still naming the same file.
    """
    assert isinstance(value, RecordValue)
    text = option_text(value, nominals=nominals)
    return None if text is None else Path(text)


def _settings_value(path: Path | None, nominals: BuiltinNominals) -> RecordValue:
    """Encode a settings-path field back into its ``Option[text]`` member value.

    The spelling is ``Path``'s normalized form, chosen deliberately over
    memoizing the original operand text; see :func:`_decode_settings_file`.
    """
    if path is None:
        return none_value(nominals=nominals)
    return some_value(TextValue(str(path)), nominals=nominals)


def _decode_bool(value: Value) -> bool:
    assert isinstance(value, BoolValue)
    return value.value
