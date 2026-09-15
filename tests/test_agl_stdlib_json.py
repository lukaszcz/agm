"""Companion contracts for the ``std/json`` standard-library module."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.boundary import AglException, AglJson, decode_boundary_value
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.semantics.values import JsonValue, RecordValue, TextValue
from tests._agl_helpers import option_nominal_descriptors

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "packages" / "stdlib"
_JSON_MODULE = ModuleId(("std", "json"))
_JSON_PARSE_ERROR = NominalId(9_400_001)
_KEY_ERROR = NominalId(9_400_002)
_OPTION = NominalId(9_400_003)
_OPTION_NONE = NominalId(9_400_004)
_OPTION_SOME = NominalId(9_400_005)


class _JsonCompanion(Protocol):
    def get(self, value: object, key: str) -> object: ...

    def get_option(self, value: object, key: str) -> object: ...


def _json_companion() -> _JsonCompanion:
    """Load ``std/json`` through the same extern boundary as production."""
    registry = ExternRegistry()
    registry.set_nominals(
        {
            _JSON_PARSE_ERROR: NominalDescriptor(
                nominal=_JSON_PARSE_ERROR,
                module_id=ModuleId(("std", "errors")),
                scope_path=(),
                declared_name="JsonParseError",
                kind=NominalKind.EXCEPTION,
                fields=("message", "raw"),
            ),
            _KEY_ERROR: NominalDescriptor(
                nominal=_KEY_ERROR,
                module_id=ModuleId(("std", "errors")),
                scope_path=(),
                declared_name="KeyError",
                kind=NominalKind.EXCEPTION,
                fields=("message", "key"),
            ),
            **option_nominal_descriptors(_OPTION, _OPTION_NONE, _OPTION_SOME),
        }
    )
    module: ModuleType = registry.load_companion(_JSON_MODULE, _STDLIB_ROOT / "src" / "json.py")
    return cast(_JsonCompanion, module)


def test_json_companion_get_handles_object_and_non_object_receivers() -> None:
    companion = _json_companion()

    assert decode_boundary_value(companion.get(AglJson({"present": 1}), "present")) == JsonValue(1)
    assert decode_boundary_value(companion.get_option(AglJson({}), "missing")) == RecordValue(
        _OPTION_NONE, "Option::None", {}
    )

    for raw in ([], True):
        with pytest.raises(AglException) as exc_info:
            companion.get(AglJson(raw), "missing")
        assert exc_info.value.value.fields["key"] == TextValue("missing")
        assert decode_boundary_value(companion.get_option(AglJson(raw), "missing")) == RecordValue(
            _OPTION_NONE, "Option::None", {}
        )
