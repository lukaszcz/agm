"""Visibility and companion contracts for the ``std/text`` standard-library module."""

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
from agm.agl.semantics.values import ArrayValue, IntValue, RecordValue, TextValue
from agm.agl.typecheck import AglTypeError
from tests._agl_helpers import option_nominal_descriptors
from tests.agl.module_graph import resolve_and_check_inline_entry

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "stdlib"
_TEXT_MODULE = ModuleId(("std", "text"))
_INDEX_ERROR = NominalId(9_300_001)
_OPTION = NominalId(9_300_002)
_OPTION_NONE = NominalId(9_300_003)
_OPTION_SOME = NominalId(9_300_004)


class _TextCompanion(Protocol):
    def chars(self, value: str) -> object: ...

    def index_of(self, value: str, substring: str) -> int: ...

    def index_of_option(self, value: str, substring: str) -> object: ...

    def pad_start(self, value: str, length: int, fill: str) -> str: ...

    def pad_end(self, value: str, length: int, fill: str) -> str: ...


def _text_companion() -> _TextCompanion:
    """Load ``std/text`` through the same extern boundary as production."""
    registry = ExternRegistry()
    registry.set_nominals(
        {
            _INDEX_ERROR: NominalDescriptor(
                nominal=_INDEX_ERROR,
                module_id=ModuleId(("std", "core")),
                scope_path=(),
                declared_name="IndexError",
                kind=NominalKind.EXCEPTION,
                fields=("message", "index", "length"),
            ),
            **option_nominal_descriptors(_OPTION, _OPTION_NONE, _OPTION_SOME),
        }
    )
    module: ModuleType = registry.load_companion(_TEXT_MODULE, _STDLIB_ROOT / "std" / "text.py")
    return cast(_TextCompanion, module)


def test_text_companion_uses_unicode_code_points_and_option_search() -> None:
    companion = _text_companion()

    assert decode_boundary_value(companion.chars("é😀")) == ArrayValue(
        [TextValue("é"), TextValue("😀")]
    )
    assert companion.index_of("banana", "na") == 2
    assert decode_boundary_value(companion.index_of_option("banana", "zz")) == RecordValue(
        _OPTION_NONE, "Option::None", {}
    )

    with pytest.raises(AglException) as exc_info:
        companion.index_of("banana", "zz")
    assert exc_info.value.value.fields["index"] == IntValue(-1)
    assert exc_info.value.value.fields["length"] == IntValue(6)


def test_text_companion_padding_counts_unicode_code_points() -> None:
    companion = _text_companion()

    assert companion.pad_start("😀", 3, "éx") == "éx😀"
    assert companion.pad_end("😀", 4, "éx") == "😀éxé"
    assert companion.pad_start("text", 2, "0") == "text"
    assert companion.pad_end("text", 8, "") == "text"


def test_text_methods_are_ambient_but_interp_requires_an_import() -> None:
    resolve_and_check_inline_entry('"value".trim()\n', HostCapabilities())

    with pytest.raises(AglScopeError):
        resolve_and_check_inline_entry('interp("value", {})\n', HostCapabilities())

    resolve_and_check_inline_entry(
        'import std/text\ntext::interp("value", {})\n', HostCapabilities()
    )

    with pytest.raises(AglTypeError, match="text is immutable"):
        resolve_and_check_inline_entry('var value = "x"\nvalue[0] := "y"\n', HostCapabilities())
