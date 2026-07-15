"""Parser-backed checks for complete AgL reference examples."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.matchcompile import compile_program_matches
from agm.agl.parser import parse_program
from agm.agl.scope import resolve
from agm.agl.typecheck import check

_REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "agl" / "reference"


def _agl_block_after(page: str, marker: str) -> str:
    text = (_REFERENCE / page).read_text()
    section = text.split(marker, maxsplit=1)[1]
    fenced = section.split("```agl\n", maxsplit=1)[1]
    return fenced.split("```", maxsplit=1)[0]


@pytest.mark.parametrize(
    "marker",
    [
        "### Constructor patterns",
        "Named sub-patterns nest arbitrarily",
    ],
)
def test_pattern_matching_complete_examples_pass_static_pipeline(marker: str) -> None:
    source = _agl_block_after("pattern-matching.md", marker)
    checked = check(resolve(parse_program(source)), HostCapabilities())
    assert compile_program_matches(checked).issues == ()
