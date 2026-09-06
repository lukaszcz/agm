"""Companion contracts for the ``std/path`` standard-library module."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

from agm.agl.ir.ids import NominalId
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.boundary import decode_boundary_value
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.semantics.values import ArrayValue, RecordValue, TextValue
from tests._agl_helpers import option_nominal_descriptors

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "stdlib"
_PATH_MODULE = ModuleId(("std", "path"))
_OPTION = NominalId(9_800_001)
_OPTION_NONE = NominalId(9_800_002)
_OPTION_SOME = NominalId(9_800_003)


class _PathCompanion(Protocol):
    def join(self, parts: list[str]) -> str: ...

    def dirname(self, path: str) -> str: ...

    def basename(self, path: str) -> str: ...

    def extension(self, path: str) -> object: ...

    def with_extension(self, path: str, extension: str) -> str: ...

    def absolute(self, path: str) -> str: ...

    def normalize(self, path: str) -> str: ...

    def relative(self, path: str, base: str) -> str: ...

    def is_absolute(self, path: str) -> bool: ...

    def parts(self, path: str) -> object: ...

    def home(self) -> str: ...


def _path_companion() -> _PathCompanion:
    registry = ExternRegistry()
    registry.set_nominals(
        {
            **option_nominal_descriptors(_OPTION, _OPTION_NONE, _OPTION_SOME),
        }
    )
    module: ModuleType = registry.load_companion(_PATH_MODULE, _STDLIB_ROOT / "src" / "path.py")
    return cast(_PathCompanion, module)


def test_path_operations_preserve_platform_path_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    companion = _path_companion()
    monkeypatch.chdir(tmp_path)

    parent = os.path.join("one", "two")
    file_path = os.path.join(parent, "file.txt")
    no_extension = os.path.join(parent, "file")
    trailing_parent = parent + os.sep

    assert companion.join([]) == ""
    assert companion.join(["one", "two", "three.txt"]) == os.path.join("one", "two", "three.txt")
    assert companion.dirname(file_path) == parent
    assert companion.basename(trailing_parent) == ""
    assert decode_boundary_value(companion.extension(file_path)) == RecordValue(
        _OPTION_SOME, "Option::Some", {"value": TextValue(".txt")}
    )
    assert decode_boundary_value(companion.extension(no_extension)) == RecordValue(
        _OPTION_NONE, "Option::None", {}
    )
    assert companion.with_extension(no_extension, ".bak") == os.path.join(parent, "file.bak")
    assert companion.absolute(os.path.join("one", "..", "two")) == str(tmp_path / "two")
    assert companion.normalize(os.path.join("one", "..", "two", ".")) == "two"
    assert companion.relative(file_path, "one") == os.path.join("two", "file.txt")
    assert not companion.is_absolute(file_path)
    assert companion.is_absolute(str(tmp_path))
    assert decode_boundary_value(companion.parts(trailing_parent)) == ArrayValue(
        [TextValue("one"), TextValue("two")]
    )
    assert companion.home() == str(Path.home())
