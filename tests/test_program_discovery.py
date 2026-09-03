"""Tests for ``program def`` declaration discovery and entry-program selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.runtime.types import ProgramDeclInfo
from tests._package_helpers import write_installed_package

# ---------------------------------------------------------------------------
# Program declaration discovery
# ---------------------------------------------------------------------------


class TestProgramDeclarationDiscovery:
    def test_discovers_program_value_parameters(self) -> None:
        from agm.cli_support.program_discovery import discover_program_declarations_from_source

        programs = discover_program_declarations_from_source(
            "program def main(name: text) -> unit = print name"
        )

        assert [program.name for program in programs] == ["main"]
        assert [param.name for param in programs[0].parameters] == ["name"]

    def test_inline_source_wraps_before_discovering_programs(self) -> None:
        from agm.cli_support.program_discovery import discover_program_declarations_from_source

        programs = discover_program_declarations_from_source(
            'let value = "inline"\nprint value', inline_source=True
        )

        assert [program.name for program in programs] == ["main"]

    def test_invalid_inline_source_degrades_to_no_programs(self) -> None:
        from agm.cli_support.program_discovery import discover_program_declarations_from_source

        assert (
            discover_program_declarations_from_source("let count: int =", inline_source=True) == ()
        )

    def test_invalid_file_source_degrades_to_no_programs(self) -> None:
        from agm.cli_support.program_discovery import discover_program_declarations_from_source

        assert discover_program_declarations_from_source("let count: int =") == ()

    def test_a_file_sources_unexpected_pipeline_exception_degrades_to_no_programs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``prepare_program`` is documented non-raising, but the file-source

        branch's ``except`` is still a real safety net, not dead code: force
        it to raise and confirm discovery degrades instead of propagating.
        """
        from agm.agl import PipelineDriver

        monkeypatch.setattr(
            PipelineDriver,
            "prepare_program",
            staticmethod(lambda source: (_ for _ in ()).throw(RuntimeError("boom"))),
        )
        from agm.cli_support.program_discovery import discover_program_declarations_from_source

        assert discover_program_declarations_from_source("program def main() -> unit = ()") == ()

    def test_unreadable_installed_entry_degrades_to_no_programs(self, tmp_path: Path) -> None:
        from agm.cli_support.exec_target import (
            PackageProgramReference,
            resolve_installed_reference,
        )
        from agm.cli_support.program_discovery import (
            discover_program_declarations_from_installed_reference,
        )

        module = write_installed_package(tmp_path, "tools")
        target = resolve_installed_reference(
            "tools/main::main", home=tmp_path, proj_dir=None, cwd=tmp_path
        )
        assert isinstance(target, PackageProgramReference)
        module.unlink()

        assert (
            discover_program_declarations_from_installed_reference(
                target, home=tmp_path, proj_dir=None, cwd=tmp_path
            )
            == ()
        )

    def test_discovers_programs_for_an_installed_reference(self, tmp_path: Path) -> None:
        from agm.cli_support.exec_target import (
            PackageProgramReference,
            resolve_installed_reference,
        )
        from agm.cli_support.program_discovery import (
            discover_program_declarations_from_installed_reference,
        )

        write_installed_package(tmp_path, "tools")
        target = resolve_installed_reference(
            "tools/main::main", home=tmp_path, proj_dir=None, cwd=tmp_path
        )
        assert isinstance(target, PackageProgramReference)

        programs = discover_program_declarations_from_installed_reference(
            target, home=tmp_path, proj_dir=None, cwd=tmp_path
        )

        assert [program.name for program in programs] == ["main"]


# ---------------------------------------------------------------------------
# select_entry_program
# ---------------------------------------------------------------------------


def _make_program(name: str, *, is_entry: bool = True) -> ProgramDeclInfo:
    """Build a minimal ``ProgramDeclInfo`` for selection tests."""
    from agm.agl.modules.ids import ENTRY_ID, ModuleId
    from agm.agl.syntax.spans import SourceSpan

    return ProgramDeclInfo(
        module=ENTRY_ID if is_entry else ModuleId(segments=("helper",)),
        scope_path=(),
        name=name,
        node_id=0,
        span=SourceSpan(1, 1, 1, 2, 0, 1),
        parameters=(),
    )


class TestSelectEntryProgram:
    def test_filters_out_a_non_entry_program(self) -> None:
        from agm.cli_support.program_discovery import select_entry_program

        entry = _make_program("main")
        imported = _make_program("helper-main", is_entry=False)

        selection = select_entry_program((entry, imported), requested=None)

        assert selection.entry_programs == (entry,)
        assert selection.selected is entry
        assert selection.requested_unmatched is False

    def test_no_entry_programs_selects_nothing(self) -> None:
        from agm.cli_support.program_discovery import select_entry_program

        selection = select_entry_program((), requested=None)

        assert selection.entry_programs == ()
        assert selection.selected is None
        assert selection.requested_unmatched is False

    def test_several_entry_programs_without_a_request_selects_none(self) -> None:
        from agm.cli_support.program_discovery import select_entry_program

        first = _make_program("first")
        second = _make_program("second")

        selection = select_entry_program((first, second), requested=None)

        assert selection.entry_programs == (first, second)
        assert selection.selected is None
        assert selection.requested_unmatched is False

    def test_a_request_matching_one_of_several_selects_it(self) -> None:
        from agm.cli_support.program_discovery import select_entry_program

        first = _make_program("first")
        second = _make_program("second")

        selection = select_entry_program((first, second), requested="second")

        assert selection.selected is second
        assert selection.requested_unmatched is False

    def test_a_request_matching_nothing_reports_unmatched(self) -> None:
        from agm.cli_support.program_discovery import select_entry_program

        entry = _make_program("main")

        selection = select_entry_program((entry,), requested="wrong")

        assert selection.entry_programs == (entry,)
        assert selection.selected is None
        assert selection.requested_unmatched is True
