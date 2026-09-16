"""Tests for ``PipelineDriver``'s ``parse_entry``/``prepare_parsed_entry`` split."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.modules.roots import RootSet
from agm.agl.parser import AglSyntaxError
from agm.agl.pipeline import ParsedEntry, PipelineDriver
from tests._agl_helpers import agl_roots, prepare_inline_command
from tests.agl.module_graph import load_graph


def _roots() -> RootSet:
    return agl_roots()


class TestParseEntry:
    def test_exposes_parsed_program_and_next_id(self) -> None:
        parsed = PipelineDriver.parse_entry("let x = 1\nx")
        assert isinstance(parsed, ParsedEntry)
        assert parsed.diagnostics == ()
        assert parsed.program is not None
        assert parsed.next_id > 0

    def test_syntax_error_is_captured_as_a_diagnostic(self) -> None:
        parsed = PipelineDriver.parse_entry("let x = (")
        assert parsed.program is None
        assert parsed.diagnostics

    def test_file_syntax_error_uses_the_same_resolved_label_as_load_graph(
        self, tmp_path: Path
    ) -> None:
        entry_path = tmp_path / "entry.agl"
        entry_path.write_text("let x = (")

        with pytest.raises(AglSyntaxError) as raised:
            load_graph("let x = (", entry_path=entry_path, roots=_roots())
        parsed = PipelineDriver.parse_entry("let x = (", entry_path=entry_path)

        assert parsed.diagnostics
        assert raised.value.to_diagnostic().source_label == parsed.diagnostics[0].source_label
        assert parsed.diagnostics[0].source_label == str(entry_path.resolve())

    def test_prepare_parsed_entry_matches_prepare_program(self) -> None:
        """``prepare_program`` is a thin wrapper: same result either way."""
        source = "import std/config::*\nprogram def main() -> unit = ()"
        parsed = PipelineDriver.parse_entry(source, entry_path=None)
        via_split = PipelineDriver.prepare_parsed_entry(parsed, roots=_roots(), default_stdlib=True)
        via_wrapper = prepare_inline_command(source, entry_path=None, roots=_roots())

        assert via_split.diagnostics == via_wrapper.diagnostics == ()
        assert via_split.resolved is not None
        assert via_wrapper.resolved is not None
