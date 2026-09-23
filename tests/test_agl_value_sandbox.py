"""Acceptance tests for value-driven AgL ``AgentSandbox``/``Sandbox`` decoding."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agent.spec import PermissionMode
from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS, BuiltinNominals
from agm.agl.runtime.sandbox_values import decode_agent_sandbox, decode_sandbox_record
from agm.agl.semantics.values import BoolValue, RecordValue, TextValue
from agm.sandbox.request import Default, SandboxLimits


def _member(nominals: BuiltinNominals, enum_name: str, member_name: str) -> RecordValue:
    return RecordValue(
        nominal=nominals.resolve_standard_member(enum_name, member_name).nominal, fields={}
    )


def _optional_default(nominals: BuiltinNominals) -> RecordValue:
    return _member(nominals, "Optional", "Default")


def _option_none(nominals: BuiltinNominals) -> RecordValue:
    return _member(nominals, "Option", "None")


def _option_some_text(nominals: BuiltinNominals, value: str) -> RecordValue:
    return RecordValue(
        nominal=nominals.resolve_standard_member("Option", "Some").nominal,
        fields={"value": TextValue(value)},
    )


def _sandbox_record(
    nominals: BuiltinNominals,
    *,
    memory: RecordValue,
    swap: RecordValue,
    settings: RecordValue,
    patch: bool,
) -> RecordValue:
    return RecordValue(
        nominal=nominals.resolve("Sandbox").nominal,
        fields={"memory": memory, "swap": swap, "settings": settings, "patch": BoolValue(patch)},
    )


def _all_default_sandbox_record(nominals: BuiltinNominals) -> RecordValue:
    return _sandbox_record(
        nominals,
        memory=_optional_default(nominals),
        swap=_optional_default(nominals),
        settings=_option_none(nominals),
        patch=True,
    )


class TestDecodeAgentSandbox:
    def test_disabled_decodes_to_none_permission_mode_and_no_limits(self) -> None:
        value = _member(NO_BUILTIN_DECLARATIONS, "AgentSandbox", "Disabled")

        mode, limits = decode_agent_sandbox(value, NO_BUILTIN_DECLARATIONS)

        assert mode is PermissionMode.NONE
        assert limits is None

    def test_native_decodes_to_native_permission_mode_and_no_limits(self) -> None:
        value = _member(NO_BUILTIN_DECLARATIONS, "AgentSandbox", "Native")

        mode, limits = decode_agent_sandbox(value, NO_BUILTIN_DECLARATIONS)

        assert mode is PermissionMode.NATIVE
        assert limits is None

    def test_sandbox_member_decodes_to_unrestricted_permission_mode_and_its_limits(self) -> None:
        value = _all_default_sandbox_record(NO_BUILTIN_DECLARATIONS)

        mode, limits = decode_agent_sandbox(value, NO_BUILTIN_DECLARATIONS)

        assert mode is PermissionMode.UNRESTRICTED
        assert limits == SandboxLimits()

    def test_member_resolution_is_by_nominal_identity_not_field_shape(self) -> None:
        """A record with the ``Sandbox`` record's exact field shape, but a
        different (fresh) identity, must not decode as the ``Sandbox`` member:
        member resolution goes through nominal identity, never by
        string/shape-matching the value itself."""
        from agm.agl.ir.ids import NominalId

        impostor = RecordValue(
            nominal=NominalId(-999_999),
            fields={
                "memory": _optional_default(NO_BUILTIN_DECLARATIONS),
                "swap": _optional_default(NO_BUILTIN_DECLARATIONS),
                "settings": _option_none(NO_BUILTIN_DECLARATIONS),
                "patch": BoolValue(True),
            },
        )

        with pytest.raises(ValueError, match="AgentSandbox"):
            decode_agent_sandbox(impostor, NO_BUILTIN_DECLARATIONS)


class TestDecodeSandboxRecord:
    def test_an_unrecognized_limit_member_is_rejected(self) -> None:
        """A ``memory``/``swap`` field carrying a record that is neither
        ``Default``, ``None``, nor ``Some`` cannot be decoded -- identity
        resolution, not field shape, drives every member match here too."""
        from agm.agl.ir.ids import NominalId

        impostor = RecordValue(nominal=NominalId(-999_999), fields={})
        record = _sandbox_record(
            NO_BUILTIN_DECLARATIONS,
            memory=impostor,
            swap=_optional_default(NO_BUILTIN_DECLARATIONS),
            settings=_option_none(NO_BUILTIN_DECLARATIONS),
            patch=True,
        )

        with pytest.raises(ValueError, match="Optional"):
            decode_sandbox_record(record, NO_BUILTIN_DECLARATIONS)

    def test_all_defaults_decode_to_the_all_defaults_limits(self) -> None:
        record = _all_default_sandbox_record(NO_BUILTIN_DECLARATIONS)

        limits = decode_sandbox_record(record, NO_BUILTIN_DECLARATIONS)

        assert limits == SandboxLimits(memory=Default, swap=Default, settings_file=None, patch=True)

    def test_explicit_memory_and_swap_text_decode_verbatim(self) -> None:
        record = _sandbox_record(
            NO_BUILTIN_DECLARATIONS,
            memory=RecordValue(
                nominal=NO_BUILTIN_DECLARATIONS.resolve_standard_member("Optional", "Some").nominal,
                fields={"value": TextValue("8G")},
            ),
            swap=RecordValue(
                nominal=NO_BUILTIN_DECLARATIONS.resolve_standard_member("Optional", "Some").nominal,
                fields={"value": TextValue("1G")},
            ),
            settings=_option_none(NO_BUILTIN_DECLARATIONS),
            patch=True,
        )

        limits = decode_sandbox_record(record, NO_BUILTIN_DECLARATIONS)

        assert limits.memory == "8G"
        assert limits.swap == "1G"

    def test_none_memory_and_swap_decode_to_unlimited(self) -> None:
        record = _sandbox_record(
            NO_BUILTIN_DECLARATIONS,
            memory=RecordValue(
                nominal=NO_BUILTIN_DECLARATIONS.resolve_standard_member("Optional", "None").nominal,
                fields={},
            ),
            swap=RecordValue(
                nominal=NO_BUILTIN_DECLARATIONS.resolve_standard_member("Optional", "None").nominal,
                fields={},
            ),
            settings=_option_none(NO_BUILTIN_DECLARATIONS),
            patch=True,
        )

        limits = decode_sandbox_record(record, NO_BUILTIN_DECLARATIONS)

        assert limits.memory is None
        assert limits.swap is None

    def test_settings_file_some_decodes_to_a_path(self) -> None:
        record = _sandbox_record(
            NO_BUILTIN_DECLARATIONS,
            memory=_optional_default(NO_BUILTIN_DECLARATIONS),
            swap=_optional_default(NO_BUILTIN_DECLARATIONS),
            settings=_option_some_text(NO_BUILTIN_DECLARATIONS, "/etc/sandbox.toml"),
            patch=False,
        )

        limits = decode_sandbox_record(record, NO_BUILTIN_DECLARATIONS)

        assert limits.settings_file == Path("/etc/sandbox.toml")
        assert limits.patch is False
