"""Package ownership and naming-discipline validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import semver

from agm.packages.discipline import DisciplineError, validate_package
from agm.packages.manifest import CommandSpec, PackageManifest, load_manifest
from agm.packages.model import PackageInfo, owning_package

FIXTURES = Path(__file__).parent / "agl" / "packages"


def _package(name: str) -> PackageInfo:
    root = FIXTURES / name
    return PackageInfo(root=root, manifest=load_manifest(root / "package.toml"))


def _custom_package(
    tmp_path: Path, *, command_path: str | None = None, program: str = "custom/main::main"
) -> PackageInfo:
    root = tmp_path / "package"
    module_root = root / "custom"
    module_root.mkdir(parents=True)
    commands = {} if command_path is None else {command_path: CommandSpec(program)}
    return PackageInfo(
        root=root,
        manifest=PackageManifest("custom", semver.Version.parse("1.0.0"), commands=commands),
    )


class TestPackageDiscipline:
    def test_accepts_well_disciplined_package(self) -> None:
        validate_package(_package("valid"))

    @pytest.mark.parametrize("fixture", ("tree_mismatch", "reserved_name"))
    def test_rejects_invalid_module_tree_or_reserved_name(self, fixture: str) -> None:
        with pytest.raises(DisciplineError):
            validate_package(_package(fixture))

    def test_rejects_unresolvable_command_program_reference(self) -> None:
        with pytest.raises(DisciplineError):
            validate_package(_package("missing_program"))

    @pytest.mark.parametrize("fixture", ("malformed_command", "reserved_command"))
    def test_rejects_malformed_or_reserved_command_path(self, fixture: str) -> None:
        with pytest.raises(DisciplineError):
            validate_package(_package(fixture))

    @pytest.mark.parametrize("alias", ("wsp", "wt", "cp", "copy"))
    def test_rejects_command_path_starting_with_a_builtin_alias(
        self, tmp_path: Path, alias: str
    ) -> None:
        package = _custom_package(tmp_path, command_path=f"{alias} launch")
        (package.module_root / "main.agl").write_text("program def main() -> unit = ()\n")

        with pytest.raises(DisciplineError):
            validate_package(package)

    def test_rejects_command_path_starting_with_an_option(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path, command_path="--launch")
        (package.module_root / "main.agl").write_text("program def main() -> unit = ()\n")

        with pytest.raises(DisciplineError):
            validate_package(package)

    @pytest.mark.parametrize(
        "reference",
        (
            "custom/main",
            "custom-/main::main",
            "custom/main::invalid/path",
            "outside/main::main",
            "custom/missing::main",
        ),
    )
    def test_rejects_malformed_or_unavailable_program_references(
        self, tmp_path: Path, reference: str
    ) -> None:
        package = _custom_package(tmp_path, command_path="start", program=reference)
        (package.module_root / "main.agl").write_text("program def main() -> unit = ()\n")

        with pytest.raises(DisciplineError):
            validate_package(package)

    def test_rejects_module_without_referenced_program(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path, command_path="start")
        (package.module_root / "main.agl").write_text("def main() -> unit = ()\n")

        with pytest.raises(DisciplineError):
            validate_package(package)

    def test_rejects_unparseable_module_and_invalid_module_filename(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path, command_path="start")
        (package.module_root / "main.agl").write_text("program def\n")

        with pytest.raises(DisciplineError):
            validate_package(package)

        (package.module_root / "main.agl").unlink()
        (package.module_root / "not-valid.agl").write_text("program def main() -> unit = ()\n")
        with pytest.raises(DisciplineError):
            validate_package(package)

    def test_ignores_non_file_agl_matches(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "cache.agl").mkdir()

        validate_package(package)

    def test_maps_only_module_tree_files_to_their_owning_package(self) -> None:
        package = _package("valid")
        module = package.module_root / "review.agl"

        assert owning_package(module.parent / "nested" / ".." / "review.agl", (package,)) == package
        assert owning_package(package.root / "prompts" / "review.md", (package,)) is None
        assert (
            owning_package(
                FIXTURES / "missing_program" / "missing_program" / "main.agl", (package,)
            )
            is None
        )

    def test_prefers_the_most_specific_nested_package(self, tmp_path: Path) -> None:
        outer = PackageInfo(
            tmp_path,
            PackageManifest("outer", semver.Version.parse("1.0.0")),
        )
        inner_root = tmp_path / "outer" / "vendor"
        inner = PackageInfo(
            inner_root,
            PackageManifest("inner", semver.Version.parse("1.0.0")),
        )
        module = inner.module_root / "main.agl"
        module.parent.mkdir(parents=True)
        module.touch()

        assert owning_package(module, (outer, inner)) == inner

    def test_maps_files_under_a_symlinked_module_tree_to_their_owning_package(
        self, tmp_path: Path
    ) -> None:
        package = _custom_package(tmp_path)
        target = tmp_path / "target"
        target.mkdir()
        package.module_root.rmdir()
        package.module_root.symlink_to(target, target_is_directory=True)
        module = target / "main.agl"
        module.write_text("program def main() -> unit = ()\n")

        assert owning_package(module, (package,)) == package
        with pytest.raises(DisciplineError, match="escapes"):
            validate_package(package)

    def test_rejects_missing_resource_through_a_scoped_using_alias(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "main.agl").write_text(
            "scope Assets\n"
            "import std/core using resource as asset, resource-dir as assets\n"
            "let root = assets()\n"
            'let prompt = asset("prompts/missing.md")\n'
            "end Assets\n"
            "program def main() -> unit = ()\n"
        )

        with pytest.raises(DisciplineError):
            validate_package(package)

    def test_rejects_invalid_utf8_referenced_source_fixture(self) -> None:
        with pytest.raises(DisciplineError):
            validate_package(_package("invalid_utf8_source"))
