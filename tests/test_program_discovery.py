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

    def test_marks_parameters_annotated_as_paths(self) -> None:
        from agm.cli_support.program_discovery import discover_program_declarations_from_source

        (program,) = discover_program_declarations_from_source(
            "import std/path\n"
            "type location = path\n"
            "type maybe-location = Option[location]\n"
            "type label = text\n"
            "\n"
            "scope files\n"
            "  type spot = path\n"
            "end files\n"
            "\n"
            "program def main(\n"
            "  plain: path,\n"
            "  qualified: path::path,\n"
            "  scoped: files::spot,\n"
            "  optional: Option[path],\n"
            "  aliased: location,\n"
            "  optional-alias: maybe-location,\n"
            "  words: text,\n"
            "  text-alias: label,\n"
            "  many: array[path],\n"
            "  optional-text: Option[text],\n"
            ") -> unit = ()\n"
        )

        assert {param.name: param.is_path for param in program.parameters} == {
            "plain": True,
            "qualified": True,
            "scoped": True,
            "optional": True,
            "aliased": True,
            "optional-alias": True,
            "words": False,
            "text-alias": False,
            "many": False,
            "optional-text": False,
        }

    def test_an_own_text_alias_named_path_is_not_a_path(self) -> None:
        from agm.cli_support.program_discovery import discover_program_declarations_from_source

        (program,) = discover_program_declarations_from_source(
            "type path = text\nprogram def main(target: path) -> unit = ()\n"
        )

        assert [param.is_path for param in program.parameters] == [False]

    def test_marks_the_reserved_path_without_the_standard_library(self) -> None:
        from agm.cli_support.program_discovery import discover_program_declarations_from_source

        (program,) = discover_program_declarations_from_source(
            "program def main(target: path, name: text) -> unit = ()\n", default_stdlib=False
        )

        assert [param.is_path for param in program.parameters] == [True, False]

    def test_follows_a_path_alias_imported_from_another_module(self, tmp_path: Path) -> None:
        from agm.cli_support.program_discovery import discover_program_declarations_from_source
        from tests._agl_helpers import agl_roots

        (tmp_path / "shared.agl").write_text(
            "scope files\n  type location = path\nend files\n", encoding="utf-8"
        )

        (program,) = discover_program_declarations_from_source(
            "import shared\n"
            "use shared::*\n"
            "program def main(\n"
            "  target: shared::files::location,\n"
            "  opened: files::location,\n"
            "  anchored: /shared::files::location,\n"
            ") -> unit = ()\n",
            roots=agl_roots(tmp_path),
        )

        assert [param.is_path for param in program.parameters] == [True, True, True]

    def test_discovers_the_option_presentation_and_documentation(self) -> None:
        from agm.agl.attributes import ProgramOptionSpec
        from agm.cli_support.program_discovery import discover_program_declarations_from_source

        programs = discover_program_declarations_from_source(
            '@doc("Greets someone.")\n'
            "program def main(\n"
            '  @opt-name("addressee") @opt-short("a") @doc("who to greet") who: text = "world",\n'
            "  plain: int = 0,\n"
            ") -> unit = print who"
        )

        (program,) = programs
        assert program.doc == "Greets someone."
        assert [param.cli for param in program.parameters] == [
            ProgramOptionSpec(name="addressee", short="a", doc="who to greet"),
            ProgramOptionSpec(name="plain"),
        ]

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
        is_entry=is_entry,
        doc=None,
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


# ---------------------------------------------------------------------------
# discover_programs_for_target
# ---------------------------------------------------------------------------


class TestDiscoverProgramsForTarget:
    def test_a_readable_file_returns_its_discovered_programs(self, tmp_path: Path) -> None:
        from agm.cli_support.program_discovery import discover_programs_for_target

        source = tmp_path / "main.agl"
        source.write_text("program def main() -> unit = ()\n", encoding="utf-8")

        programs, referenced = discover_programs_for_target(
            file=str(source), command=None, module_paths=None, no_stdlib=True
        )

        assert tuple(program.name for program in programs) == ("main",)
        assert referenced is None

    def test_an_unreadable_file_token_degrades_silently(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The tail split probes ordinary non-option tokens as candidate FILE

        tokens, so a token that names no readable file must degrade to no
        programs without writing anything a caller cannot suppress.
        """
        from agm.cli_support.program_discovery import discover_programs_for_target

        monkeypatch.chdir(tmp_path)

        assert discover_programs_for_target(
            file="World", command=None, module_paths=None, no_stdlib=True
        ) == ((), None)

        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == ""
