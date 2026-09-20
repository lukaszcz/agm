"""Companion contracts for the ``std/path`` standard-library module."""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.boundary import AglException, decode_boundary_value
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.semantics.values import ArrayValue, RecordValue, TextValue
from tests._agl_helpers import option_nominal_descriptors

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "packages" / "stdlib"
_PATH_MODULE = ModuleId(("std", "path"))
_ERRORS_MODULE = ModuleId(("std", "errors"))
_OPTION = NominalId(9_800_001)
_OPTION_NONE = NominalId(9_800_002)
_OPTION_SOME = NominalId(9_800_003)
_ENCODING_ERROR = NominalId(9_800_004)


class _PathCompanion(Protocol):
    def join(self, parts: list[str]) -> str: ...

    def dirname(self, path: str) -> str: ...

    def basename(self, path: str) -> str: ...

    def stem(self, path: str) -> str: ...

    def extension(self, path: str) -> object: ...

    def with_extension(self, path: str, extension: str) -> str: ...

    def with_name(self, path: str, name: str) -> str: ...

    def absolute(self, path: str) -> str: ...

    def normalize(self, path: str) -> str: ...

    def relative(self, path: str, base: str) -> str: ...

    def is_absolute(self, path: str) -> bool: ...

    def is_under(self, path: str, base: str) -> bool: ...

    def parts(self, path: str) -> object: ...

    def expand_user(self, path: str) -> str: ...

    def common_prefix(self, paths: list[str]) -> object: ...

    def home(self) -> str: ...


def _path_companion() -> _PathCompanion:
    registry = ExternRegistry()
    registry.set_nominals(
        {
            **option_nominal_descriptors(_OPTION, _OPTION_NONE, _OPTION_SOME),
            _ENCODING_ERROR: NominalDescriptor(
                nominal=_ENCODING_ERROR,
                module_id=_ERRORS_MODULE,
                scope_path=(),
                declared_name="EncodingError",
                kind=NominalKind.EXCEPTION,
                fields=("message", "raw"),
            ),
        },
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
        _OPTION_SOME, {"value": TextValue(".txt")}
    )
    assert decode_boundary_value(companion.extension(no_extension)) == RecordValue(_OPTION_NONE, {})
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


def test_path_decomposition_splits_a_name_into_its_stem_and_extension() -> None:
    companion = _path_companion()

    nested = os.path.join("one", "two", "report.txt")
    assert companion.stem(nested) == "report"
    assert companion.stem(os.path.join("one", "archive.tar.gz")) == "archive.tar"
    assert companion.stem(os.path.join("one", "plain")) == "plain"
    assert companion.stem(".bashrc") == ".bashrc"
    assert companion.stem("") == ""
    # stem and extension reconstruct the basename they decomposed.
    extension = decode_boundary_value(companion.extension(nested))
    assert extension == RecordValue(_OPTION_SOME, {"value": TextValue(".txt")})
    assert companion.stem(nested) + ".txt" == companion.basename(nested)


def test_with_name_replaces_only_the_final_component() -> None:
    companion = _path_companion()

    nested = os.path.join("one", "two", "report.txt")
    assert companion.with_name(nested, "summary.md") == os.path.join("one", "two", "summary.md")
    # A bare name has no directory to preserve.
    assert companion.with_name("report.txt", "summary.md") == "summary.md"
    # An absolute root keeps its root when its (empty) name is replaced.
    assert companion.with_name(os.sep, "summary.md") == os.path.join(os.sep, "summary.md")


def test_expand_user_resolves_a_leading_tilde_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    companion = _path_companion()
    monkeypatch.setenv("HOME", str(tmp_path))

    assert companion.expand_user(os.path.join("~", "notes")) == str(tmp_path / "notes")
    assert companion.expand_user("~") == str(tmp_path)
    # A tilde anywhere but the front is an ordinary character.
    assert companion.expand_user(os.path.join("work", "~", "notes")) == os.path.join(
        "work", "~", "notes"
    )
    assert companion.expand_user(os.path.join("one", "two")) == os.path.join("one", "two")


def test_is_under_tests_containment_after_normalizing_both_operands() -> None:
    companion = _path_companion()

    assert companion.is_under(os.path.join("one", "two", "file.txt"), "one")
    assert companion.is_under(os.path.join("one", "two"), os.path.join("one", "two"))
    # Normalization happens first, so traversal cannot escape unnoticed.
    assert not companion.is_under(os.path.join("one", "two", "..", ".."), "one")
    assert companion.is_under(os.path.join("one", "two", "..", "three"), "one")
    assert not companion.is_under("one", os.path.join("one", "two"))
    assert not companion.is_under("onetwo", "one")
    # Mixing an absolute path with a relative base is never containment.
    assert not companion.is_under(os.path.join(os.sep, "one", "two"), "one")


def test_common_prefix_returns_none_when_no_single_prefix_exists() -> None:
    companion = _path_companion()

    shared = [os.path.join("one", "two", "a.txt"), os.path.join("one", "three", "b.txt")]
    assert decode_boundary_value(companion.common_prefix(shared)) == RecordValue(
        _OPTION_SOME, {"value": TextValue("one")}
    )
    single = [os.path.join("one", "two")]
    assert decode_boundary_value(companion.common_prefix(single)) == RecordValue(
        _OPTION_SOME, {"value": TextValue(os.path.join("one", "two"))}
    )
    none = RecordValue(_OPTION_NONE, {})
    # No paths at all, and absolute mixed with relative, have no common prefix.
    assert decode_boundary_value(companion.common_prefix([])) == none
    assert decode_boundary_value(companion.common_prefix([os.sep + "one", "one"])) == none


def test_absolute_raises_encoding_error_when_the_working_directory_is_not_valid_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companion = _path_companion()
    monkeypatch.setattr(os, "getcwd", lambda: "/tmp/h\udcffome")

    with pytest.raises(AglException) as exc_info:
        companion.absolute("x")
    assert exc_info.value.value.nominal == _ENCODING_ERROR
    assert exc_info.value.value.fields["raw"] == TextValue("/tmp/h\\udcffome/x")


def test_relative_raises_encoding_error_when_the_result_is_not_valid_unicode() -> None:
    companion = _path_companion()

    with pytest.raises(AglException) as exc_info:
        companion.relative("/a\udcffb/c", "/x")
    assert exc_info.value.value.nominal == _ENCODING_ERROR


def test_expand_user_raises_encoding_error_when_home_is_not_valid_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companion = _path_companion()
    monkeypatch.setenv("HOME", "/tmp/h\udcffome")

    with pytest.raises(AglException) as exc_info:
        companion.expand_user("~")
    assert exc_info.value.value.nominal == _ENCODING_ERROR
    assert exc_info.value.value.fields["raw"] == TextValue("/tmp/h\\udcffome")


def test_home_raises_encoding_error_when_the_users_home_directory_is_not_valid_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companion = _path_companion()
    monkeypatch.setenv("HOME", "/tmp/h\udcffome")

    with pytest.raises(AglException) as exc_info:
        companion.home()
    assert exc_info.value.value.nominal == _ENCODING_ERROR
    assert exc_info.value.value.fields["raw"] == TextValue("/tmp/h\\udcffome")
