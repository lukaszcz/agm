"""Companion contracts for the ``std/regex`` standard-library module."""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast
from unittest.mock import Mock

import pytest

from agm.agl.ir.ids import NominalId
from agm.agl.ir.program import NominalDescriptor, NominalKind, VariantDescriptor
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.boundary import AglException, decode_boundary_value
from agm.agl.runtime.externs import ExternRegistry
from agm.agl.semantics.values import (
    ArrayValue,
    DictValue,
    EnumValue,
    IntValue,
    RecordValue,
    TextValue,
)

_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "stdlib"
_REGEX_MODULE = ModuleId(("std", "regex"))
_MATCH = NominalId(9_700_001)
_REGEX_ERROR = NominalId(9_700_002)
_OPTION = NominalId(9_700_003)


class _RegexCompanion(Protocol):
    def test(self, pattern: str, s: str) -> bool: ...

    def find_option(self, pattern: str, s: str) -> object: ...

    def find_all(self, pattern: str, s: str) -> object: ...

    def replace(self, pattern: str, s: str, replacement: str) -> str: ...

    def split(self, pattern: str, s: str) -> object: ...

    def escape(self, s: str) -> str: ...


def _regex_companion() -> tuple[_RegexCompanion, ModuleType]:
    """Load ``std/regex`` through the production extern boundary."""
    registry = ExternRegistry()
    registry.set_nominals(
        {
            _MATCH: NominalDescriptor(
                nominal=_MATCH,
                module_id=_REGEX_MODULE,
                scope_path=(),
                declared_name="Match",
                kind=NominalKind.RECORD,
                fields=("matched", "start", "end", "groups", "named-groups"),
            ),
            _REGEX_ERROR: NominalDescriptor(
                nominal=_REGEX_ERROR,
                module_id=_REGEX_MODULE,
                scope_path=(),
                declared_name="RegexError",
                kind=NominalKind.EXCEPTION,
                fields=("message", "pattern"),
            ),
            _OPTION: NominalDescriptor(
                nominal=_OPTION,
                module_id=ModuleId(("std", "option")),
                scope_path=(),
                declared_name="Option",
                kind=NominalKind.ENUM,
                variants=(VariantDescriptor("Some", ("value",)), VariantDescriptor("None", ())),
            ),
        }
    )
    module = registry.load_companion(_REGEX_MODULE, _STDLIB_ROOT / "std" / "regex.py")
    return cast(_RegexCompanion, module), module


def test_regex_match_populates_offsets_numbered_groups_named_groups_and_nonparticipants() -> None:
    companion, _ = _regex_companion()

    found = decode_boundary_value(
        companion.find_option(r"(?P<word>[A-Za-z]+)-(\d+)(?:-([A-Z]+))?", "ref-42")
    )

    assert found == EnumValue(
        _OPTION,
        "Option",
        "Some",
        {
            "value": RecordValue(
                _MATCH,
                "Match",
                {
                    "matched": TextValue("ref-42"),
                    "start": IntValue(0),
                    "end": IntValue(6),
                    "groups": ArrayValue(
                        [
                            EnumValue(_OPTION, "Option", "Some", {"value": TextValue("ref")}),
                            EnumValue(_OPTION, "Option", "Some", {"value": TextValue("42")}),
                            EnumValue(_OPTION, "Option", "None", {}),
                        ]
                    ),
                    "named-groups": DictValue({"word": TextValue("ref")}),
                },
            )
        },
    )
    assert decode_boundary_value(companion.find_option("x", "no match")) == EnumValue(
        _OPTION, "Option", "None", {}
    )


def test_regex_find_all_replacement_split_and_escape_follow_python_re() -> None:
    companion, _ = _regex_companion()

    all_matches = decode_boundary_value(companion.find_all("[A-Za-z]+", "one 22 two"))
    assert isinstance(all_matches, ArrayValue)
    assert [match.fields["matched"] for match in all_matches.elements] == [
        TextValue("one"),
        TextValue("two"),
    ]
    assert companion.replace(r"(?P<word>[A-Za-z]+)-(\d+)", "item-42", r"\g<word>[\2]") == "item[42]"
    assert decode_boundary_value(companion.split("([,;])", "a,b;c,")) == ArrayValue(
        [
            TextValue("a"),
            TextValue(","),
            TextValue("b"),
            TextValue(";"),
            TextValue("c"),
            TextValue(","),
            TextValue(""),
        ]
    )
    escaped = companion.escape("a+b?.")
    assert companion.test(escaped, "a+b?.")


def test_regex_invalid_pattern_raises_typed_error_and_compiles_each_pattern_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    companion, module = _regex_companion()
    compile_pattern = cast(object, getattr(module, "_compile"))
    compile_pattern.cache_clear()
    compile_mock = Mock(wraps=re.compile)
    monkeypatch.setattr(module.re, "compile", compile_mock)

    assert companion.test("[0-9]+", "42")
    assert decode_boundary_value(companion.find_option("[0-9]+", "x7")) == EnumValue(
        _OPTION,
        "Option",
        "Some",
        {
            "value": RecordValue(
                _MATCH,
                "Match",
                {
                    "matched": TextValue("7"),
                    "start": IntValue(1),
                    "end": IntValue(2),
                    "groups": ArrayValue([]),
                    "named-groups": DictValue({}),
                },
            )
        },
    )
    assert compile_mock.call_count == 1

    with pytest.raises(AglException) as exc_info:
        companion.test("[", "anything")
    assert exc_info.value.value.nominal == _REGEX_ERROR
    assert exc_info.value.value.fields["pattern"] == TextValue("[")
