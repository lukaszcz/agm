"""Tests for the shared ``agm exec`` argument classifier and resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.cli_support.exec_target import (
    ExecTargetError,
    FileEntry,
    InlineSource,
    PackageProgramReference,
    is_installed_reference,
    resolve_exec_target,
    resolve_installed_reference,
)
from tests._package_helpers import write_installed_package


class TestIsInstalledReference:
    """The single classification rule shared by execution, help, and completion."""

    def test_inline_command_is_never_a_reference(self) -> None:
        assert is_installed_reference("pkg/mod::main", command="print 1") is False

    def test_missing_file_argument_is_not_a_reference(self) -> None:
        assert is_installed_reference(None, command=None) is False

    def test_no_separator_is_not_a_reference(self, tmp_path: Path) -> None:
        assert is_installed_reference(str(tmp_path / "missing.agl"), command=None) is False

    def test_missing_file_with_separator_is_a_reference(self, tmp_path: Path) -> None:
        assert is_installed_reference(str(tmp_path / "pkg/mod::main"), command=None) is True

    def test_an_existing_on_disk_file_wins_even_with_a_separator_in_its_name(
        self, tmp_path: Path
    ) -> None:
        """An on-disk file always wins classification, regardless of ``::`` in
        its name — this is the rule execution uses, now also used by
        ``--help`` and shell completion so all three agree."""
        colliding = tmp_path / "pkg::mod.agl"
        colliding.write_text("program def main() -> unit = ()\n", encoding="utf-8")

        assert is_installed_reference(str(colliding), command=None) is False


class TestResolveExecTarget:
    def test_inline_command_resolves_to_inline_source(self, tmp_path: Path) -> None:
        target = resolve_exec_target(
            file=None, command="print 1", home=tmp_path, proj_dir=None, cwd=tmp_path
        )
        assert target == InlineSource()

    def test_neither_file_nor_command_is_an_error(self, tmp_path: Path) -> None:
        target = resolve_exec_target(
            file=None, command=None, home=tmp_path, proj_dir=None, cwd=tmp_path
        )
        assert isinstance(target, ExecTargetError)

    def test_on_disk_file_resolves_to_a_file_entry(self, tmp_path: Path) -> None:
        agl_file = tmp_path / "prog.agl"
        agl_file.write_text("program def main() -> unit = ()\n", encoding="utf-8")

        target = resolve_exec_target(
            file=str(agl_file), command=None, home=tmp_path, proj_dir=None, cwd=tmp_path
        )

        assert target == FileEntry(agl_file)

    def test_installed_reference_resolves_to_its_package_and_entry(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        module = write_installed_package(home, "tools")

        target = resolve_exec_target(
            file="tools/main::main",
            command=None,
            home=home,
            proj_dir=None,
            cwd=tmp_path,
        )

        assert isinstance(target, PackageProgramReference)
        assert target.entry_path == module
        assert target.package.manifest.name == "tools"
        assert target.declaration_path == "main"


class TestResolveInstalledReference:
    def test_malformed_reference_is_an_error(self, tmp_path: Path) -> None:
        result = resolve_installed_reference(
            "not-a-reference", home=tmp_path, proj_dir=None, cwd=tmp_path
        )
        assert isinstance(result, ExecTargetError)

    def test_invalid_module_path_is_an_error(self, tmp_path: Path) -> None:
        result = resolve_installed_reference(
            "bad-name/main::main", home=tmp_path, proj_dir=None, cwd=tmp_path
        )
        assert isinstance(result, ExecTargetError)

    def test_unknown_package_is_an_error(self, tmp_path: Path) -> None:
        result = resolve_installed_reference(
            "missing/main::main", home=tmp_path, proj_dir=None, cwd=tmp_path
        )
        assert isinstance(result, ExecTargetError)
        assert "missing/main::main" in result.message

    def test_package_name_override_selects_by_the_given_name(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        write_installed_package(home, "tools")

        result = resolve_installed_reference(
            "tools/main::main", home=home, proj_dir=None, cwd=tmp_path, package_name="other"
        )

        assert isinstance(result, ExecTargetError)
        # The very same reference resolves without the override, so the
        # override — not the reference — chose the package that is missing.
        assert isinstance(
            resolve_installed_reference("tools/main::main", home=home, proj_dir=None, cwd=tmp_path),
            PackageProgramReference,
        )

    def test_broken_active_package_selection_is_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agm.cli_support.exec_target as exec_target

        monkeypatch.setattr(
            exec_target,
            "select_active_packages",
            lambda **_: (_ for _ in ()).throw(ValueError("broken")),
        )

        result = resolve_installed_reference(
            "tools/main::main", home=tmp_path, proj_dir=None, cwd=tmp_path
        )

        assert isinstance(result, ExecTargetError)
        # The underlying selection failure is propagated, not swallowed.
        assert "broken" in result.message
