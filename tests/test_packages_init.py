"""Tests for ``agm pkg init`` package scaffolding.

Covers:
- CLI wiring of the optional directory argument and the --name/--version options.
- The scaffolded directory is a valid package: ``agm pkg check`` accepts it.
- The target directory is created when it does not exist.
- An existing manifest is never overwritten, and an existing starter module is kept.
- Invalid, reserved, and malformed names and versions are refused before anything
  is written.
- --dry-run reports the scaffold without creating it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
import agm.commands.pkg.init as init_command
from agm.cli_support.args import PkgInitArgs

_AGM_COMMAND = get_command(cli.app)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def invoke(runner: CliRunner, argv: list[str]) -> Result:
    return runner.invoke(_AGM_COMMAND, argv, prog_name="agm")


class TestPackageInitParsing:
    def test_directory_name_and_version_map_to_typed_arguments(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[PkgInitArgs] = []
        monkeypatch.setattr(init_command, "run", calls.append)

        result = invoke(runner, ["pkg", "init", "demo", "--name", "tools", "--version", "2.1.0"])

        assert result.exit_code == 0
        assert calls == [PkgInitArgs(directory="demo", name="tools", version="2.1.0")]

    def test_defaults_to_the_current_directory_and_an_initial_version(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[PkgInitArgs] = []
        monkeypatch.setattr(init_command, "run", calls.append)

        result = invoke(runner, ["pkg", "init"])

        assert result.exit_code == 0
        assert calls == [PkgInitArgs(directory=None, name=None, version="0.1.0")]


class TestPackageInit:
    def test_initializes_the_current_directory_into_a_checkable_package(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        package = tmp_path / "demo"
        package.mkdir()
        monkeypatch.chdir(package)

        created = invoke(runner, ["pkg", "init"])
        checked = invoke(runner, ["pkg", "check"])

        assert created.exit_code == 0
        assert checked.exit_code == 0
        assert 'name = "demo"' in (package / "package.toml").read_text(encoding="utf-8")
        assert 'version = "0.1.0"' in (package / "package.toml").read_text(encoding="utf-8")
        assert (package / "demo" / "main.agl").read_text(encoding="utf-8")
        assert str(package / "package.toml") in created.output

    def test_manifest_documents_how_to_declare_dependencies(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        package = tmp_path / "demo"
        package.mkdir()
        monkeypatch.chdir(package)

        assert invoke(runner, ["pkg", "init"]).exit_code == 0

        manifest = (package / "package.toml").read_text(encoding="utf-8")
        assert "# [dependencies]" in manifest
        assert invoke(runner, ["pkg", "check"]).exit_code == 0

    def test_creates_a_missing_target_directory(self, runner: CliRunner, tmp_path: Path) -> None:
        result = invoke(runner, ["pkg", "init", str(tmp_path / "demo")])

        assert result.exit_code == 0
        assert (tmp_path / "demo" / "package.toml").is_file()
        assert (tmp_path / "demo" / "demo" / "main.agl").is_file()

    def test_scaffolds_a_kebab_case_package_name(self, runner: CliRunner, tmp_path: Path) -> None:
        """Package names are AgL names, so a kebab-case directory scaffolds."""
        package = tmp_path / "review-tools"

        created = invoke(runner, ["pkg", "init", str(package)])
        checked = invoke(runner, ["pkg", "check", str(package)])

        assert created.exit_code == 0
        assert checked.exit_code == 0
        assert 'name = "review-tools"' in (package / "package.toml").read_text(encoding="utf-8")
        assert (package / "review-tools" / "main.agl").is_file()

    def test_name_and_version_options_override_the_defaults(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        result = invoke(
            runner,
            [
                "pkg",
                "init",
                str(tmp_path / "review-tools"),
                "--name",
                "tools",
                "--version",
                "2.1.0",
            ],
        )

        manifest = (tmp_path / "review-tools" / "package.toml").read_text(encoding="utf-8")
        assert result.exit_code == 0
        assert 'name = "tools"' in manifest
        assert 'version = "2.1.0"' in manifest
        assert (tmp_path / "review-tools" / "tools" / "main.agl").is_file()

    def test_keeps_an_existing_starter_module(self, runner: CliRunner, tmp_path: Path) -> None:
        package = tmp_path / "demo"
        (package / "demo").mkdir(parents=True)
        (package / "demo" / "main.agl").write_text(
            "program def main() -> unit = print('hi')\n", encoding="utf-8"
        )

        result = invoke(runner, ["pkg", "init", str(package)])

        assert result.exit_code == 0
        assert "print" in (package / "demo" / "main.agl").read_text(encoding="utf-8")
        assert str(package / "demo" / "main.agl") not in result.output

    def test_refuses_a_directory_that_already_holds_a_manifest(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        package = tmp_path / "demo"
        package.mkdir()
        (package / "package.toml").write_text('[package]\nname = "other"\n', encoding="utf-8")

        result = invoke(runner, ["pkg", "init", str(package)])

        assert result.exit_code == 1
        assert result.stderr
        assert 'name = "other"' in (package / "package.toml").read_text(encoding="utf-8")
        assert not (package / "demo").exists()

    @pytest.mark.parametrize(
        "argv",
        (
            ["pkg", "init", "1review"],
            ["pkg", "init", "demo", "--name", "def"],
            ["pkg", "init", "demo", "--name", "run"],
            ["pkg", "init", "demo", "--version", "1.0"],
        ),
    )
    def test_refuses_an_invalid_name_or_version_without_writing(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)

        result = invoke(runner, argv)

        assert result.exit_code == 1
        assert result.stderr
        assert list(tmp_path.iterdir()) == []

    def test_reports_a_directory_that_cannot_be_created(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        blocker = tmp_path / "demo"
        blocker.write_text("not a directory\n", encoding="utf-8")

        result = invoke(runner, ["pkg", "init", str(blocker)])

        assert result.exit_code == 1
        assert result.stderr

    def test_dry_run_reports_the_scaffold_without_writing_it(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        result = invoke(runner, ["pkg", "init", str(tmp_path / "demo"), "--dry-run"])

        assert result.exit_code == 0
        assert "dry-run" in result.output
        assert not (tmp_path / "demo").exists()
