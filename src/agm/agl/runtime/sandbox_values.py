"""Value-driven AgL ``AgentSandbox``/``Sandbox`` decoding, beside ``agents.py``.

The value<->host boundary for the ``sandbox`` named parameter threaded
through ``ask``/``ask-request`` and their receiver forms: decodes the AgL
``AgentSandbox`` value a call site supplies into the host
``agent.spec.PermissionMode`` dispatch already understands, plus the
profile-independent ``SandboxLimits`` a ``Sandbox`` member carries. Member
identity is resolved through ``resolve_standard_member_name`` (nominal
identity), never by reading a name off the value itself, exactly as
``agents.py`` resolves an ``Agent`` member.
"""

from __future__ import annotations

from pathlib import Path

from agm.agent.spec import PermissionMode
from agm.agl.ir.builtin_nominals import BuiltinNominals, resolve_standard_member_name
from agm.agl.ir.reserved_nominals import AGENT_SANDBOX_MEMBERS
from agm.agl.runtime.option import option_text
from agm.agl.semantics.values import BoolValue, RecordValue, TextValue, Value
from agm.sandbox.request import Default, LimitSpec, SandboxLimits

__all__ = [
    "decode_agent_sandbox",
    "decode_sandbox_record",
]

_OPTIONAL_MEMBERS = ("Some", "None", "Default")


def decode_agent_sandbox(
    value: RecordValue, nominals: BuiltinNominals
) -> tuple[PermissionMode, SandboxLimits | None]:
    """Decode an ``AgentSandbox`` member value into a permission mode and optional limits.

    ``Disabled`` and ``Native`` carry no limits; the referenced ``Sandbox``
    member decodes its own record through :func:`decode_sandbox_record`.
    """
    case = resolve_standard_member_name(
        value.nominal, "AgentSandbox", AGENT_SANDBOX_MEMBERS, nominals
    )
    if case == "Disabled":
        return PermissionMode.NONE, None
    if case == "Native":
        return PermissionMode.NATIVE, None
    if case == "Sandbox":
        return PermissionMode.UNRESTRICTED, decode_sandbox_record(value, nominals)
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


def _decode_settings_file(value: Value, nominals: BuiltinNominals) -> Path | None:
    """Decode an ``Option[text]`` settings-path field."""
    assert isinstance(value, RecordValue)
    text = option_text(value, nominals=nominals)
    return None if text is None else Path(text)


def _decode_bool(value: Value) -> bool:
    assert isinstance(value, BoolValue)
    return value.value
