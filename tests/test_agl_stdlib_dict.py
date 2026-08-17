"""Visibility and companion contracts for the ``std/dict`` standard-library module."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind, VariantDescriptor
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.boundary import AglDictView, decode_boundary_value
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.scope import AglScopeError
from agm.agl.semantics.values import DictValue, EnumValue, IntValue
from tests.agl.module_graph import resolve_and_check_inline_entry

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "stdlib"
_DICT_MODULE = ModuleId(("std", "dict"))
_KEY_ERROR = NominalId(9_200_001)
_OPTION = NominalId(9_200_002)
_PAIR = NominalId(9_200_003)


class _DictCompanion(Protocol):
    def get_option(self, values: object, key: str) -> object: ...

    def remove_option(self, values: object, key: str) -> object: ...

    def set(self, values: object, key: str, value: object) -> None: ...

    def merge(self, values: object, other: object) -> object: ...

    def merge_in_place(self, values: object, other: object) -> None: ...

    def filter_in_place(self, values: object, predicate: object) -> None: ...

    def clear(self, values: object) -> None: ...


def _entries(values: object) -> dict[str, object]:
    assert isinstance(values, AglDictView)
    return dict(values)


def _exclude_two(key: str, value: object) -> bool:
    return key != "two"


def _dict_companion() -> _DictCompanion:
    """Load ``std/dict`` through the same extern boundary as production."""
    registry = ExternRegistry()
    registry.set_nominals(
        {
            _KEY_ERROR: NominalDescriptor(
                nominal=_KEY_ERROR,
                module_id=ModuleId(("std", "core")),
                scope_path=(),
                declared_name="KeyError",
                kind=NominalKind.EXCEPTION,
                fields=("message", "key"),
            ),
            _OPTION: NominalDescriptor(
                nominal=_OPTION,
                module_id=ModuleId(("std", "option")),
                scope_path=(),
                declared_name="Option",
                kind=NominalKind.ENUM,
                variants=(VariantDescriptor("Some", ("value",)), VariantDescriptor("None", ())),
            ),
            _PAIR: NominalDescriptor(
                nominal=_PAIR,
                module_id=ModuleId(("std", "pair")),
                scope_path=(),
                declared_name="Pair",
                kind=NominalKind.RECORD,
                fields=("first", "second"),
            ),
        }
    )
    module: ModuleType = registry.load_companion(_DICT_MODULE, _STDLIB_ROOT / "std" / "dict.py")
    return cast(_DictCompanion, module)


def test_dict_companion_get_and_remove_options_preserve_the_live_dict() -> None:
    companion = _dict_companion()
    values = AglDictView(DictValue({"one": IntValue(1)}))

    assert decode_boundary_value(companion.get_option(values, "one")) == EnumValue(
        _OPTION, "Option", "Some", {"value": IntValue(1)}
    )
    assert decode_boundary_value(companion.get_option(values, "missing")) == EnumValue(
        _OPTION, "Option", "None", {}
    )
    assert decode_boundary_value(companion.remove_option(values, "one")) == EnumValue(
        _OPTION, "Option", "Some", {"value": IntValue(1)}
    )
    assert decode_boundary_value(companion.remove_option(values, "missing")) == EnumValue(
        _OPTION, "Option", "None", {}
    )
    assert _entries(values) == {}


def test_dict_companion_set_updates_the_live_dict() -> None:
    companion = _dict_companion()
    values = AglDictView(DictValue({"one": IntValue(1)}))

    companion.set(values, "two", 2)
    companion.set(values, "one", 10)

    assert _entries(values) == {"one": 10, "two": 2}


def test_dict_companion_mutating_operations_update_boundary_views() -> None:
    companion = _dict_companion()
    values = AglDictView(DictValue({"one": IntValue(1), "two": IntValue(2)}))
    alias = values

    merged = companion.merge(values, AglDictView(DictValue({"two": IntValue(20)})))
    assert _entries(merged) == {"one": 1, "two": 20}
    assert _entries(values) == {"one": 1, "two": 2}

    companion.merge_in_place(values, AglDictView(DictValue({"three": IntValue(3)})))
    companion.filter_in_place(values, _exclude_two)
    companion.clear(values)

    assert _entries(alias) == {}


def test_dict_methods_are_ambient_but_from_entries_requires_an_import() -> None:
    resolve_and_check_inline_entry(
        'let values = {"one": 1}\nvalues.get("one")\n',
        HostCapabilities(),
    )

    with pytest.raises(AglScopeError):
        resolve_and_check_inline_entry("from-entries([])\n", HostCapabilities())

    resolve_and_check_inline_entry(
        'import std/dict\ndict::from-entries([Pair(first = "one", second = 1)])\n',
        HostCapabilities(),
    )
