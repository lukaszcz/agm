"""Visibility and companion contracts for the ``std/array`` standard-library module."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.boundary import AglException, decode_boundary_value
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.scope import AglScopeError
from agm.agl.semantics.values import IntValue, RecordValue
from agm.agl.typecheck import AglTypeError
from tests._agl_helpers import option_nominal_descriptors
from tests.agl.module_graph import resolve_and_check_inline_entry

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "packages" / "stdlib"
_ARRAY_MODULE = ModuleId(("std", "array"))
_INDEX_ERROR = NominalId(9_400_001)
_OPTION = NominalId(9_400_002)
_OPTION_NONE = NominalId(9_400_003)
_OPTION_SOME = NominalId(9_400_004)
_PAIR = NominalId(9_400_005)


class _ArrayCompanion(Protocol):
    def index_of(self, values: object, value: object) -> int: ...

    def index_of_option(self, values: object, value: object) -> object: ...


def _array_companion() -> _ArrayCompanion:
    """Load ``std/array`` through the same extern boundary as production."""
    registry = ExternRegistry()
    registry.set_nominals(
        {
            _INDEX_ERROR: NominalDescriptor(
                nominal=_INDEX_ERROR,
                module_id=ModuleId(("std", "errors")),
                scope_path=(),
                declared_name="IndexError",
                kind=NominalKind.EXCEPTION,
                fields=("message", "index", "length"),
            ),
            **option_nominal_descriptors(_OPTION, _OPTION_NONE, _OPTION_SOME),
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
    module: ModuleType = registry.load_companion(_ARRAY_MODULE, _STDLIB_ROOT / "src" / "array.py")
    return cast(_ArrayCompanion, module)


def test_array_methods_are_ambient_but_free_functions_require_an_import() -> None:
    resolve_and_check_inline_entry(
        "let values = [1, 2]\nvalues.map(fn(value: int) => value + 1)\n",
        HostCapabilities(),
    )

    with pytest.raises(AglScopeError):
        resolve_and_check_inline_entry("range(1, 2)\n", HostCapabilities())

    resolve_and_check_inline_entry("import std/array\narray::range(1, 2)\n", HostCapabilities())


def test_map_in_place_preserves_the_receiver_element_type() -> None:
    resolve_and_check_inline_entry(
        "let values = [1, 2]\nvalues.map!(fn(value: int) => value + 1)\n",
        HostCapabilities(),
    )

    with pytest.raises(AglTypeError):
        resolve_and_check_inline_entry(
            "let values = [1, 2]\nvalues.map!(fn(value: int) => value as text)\n",
            HostCapabilities(),
        )


def test_array_search_raises_for_an_absent_value_and_options_it_separately() -> None:
    """``index-of`` carries no ``?``, so an absent value is a typed failure —
    the same contract ``text::index-of`` follows — while ``index-of?`` reports
    the miss as ``Option::None``."""
    companion = _array_companion()
    values = [1, 2, 3]

    assert companion.index_of(values, 2) == 1
    assert decode_boundary_value(companion.index_of_option(values, 9)) == RecordValue(
        _OPTION_NONE, "Option::None", {}
    )

    with pytest.raises(AglException) as exc_info:
        companion.index_of(values, 9)
    assert exc_info.value.value.fields["index"] == IntValue(-1)
    assert exc_info.value.value.fields["length"] == IntValue(3)
