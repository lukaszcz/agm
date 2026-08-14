"""Package ownership and naming-discipline validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import semver

from agm.packages.discipline import (
    DisciplineError,
    validate_archive_package,
    validate_package,
)
from agm.packages.manifest import CommandSpec, DependencySpec, PackageManifest, load_manifest
from agm.packages.model import PackageInfo, owning_package
from tests._timeouts import fail_if_slow

FIXTURES = Path(__file__).parent / "agl" / "packages"


def _package(name: str) -> PackageInfo:
    root = FIXTURES / name
    return PackageInfo(root=root, manifest=load_manifest(root / "package.toml"))


def _custom_package(
    tmp_path: Path,
    *,
    command_path: str | None = None,
    program: str = "custom/main::main",
    dependencies: dict[str, DependencySpec] | None = None,
) -> PackageInfo:
    root = tmp_path / "package"
    module_root = root / "custom"
    module_root.mkdir(parents=True)
    commands = {} if command_path is None else {command_path: CommandSpec(program)}
    return PackageInfo(
        root=root,
        manifest=PackageManifest(
            "custom",
            semver.Version.parse("1.0.0"),
            commands=commands,
            dependencies=dependencies or {},
        ),
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

    def test_rejects_import_of_missing_package_module(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "main.agl").write_text("import custom/missing\n")

        with pytest.raises(DisciplineError):
            validate_package(package)

    def test_rejects_undeclared_import_from_mounted_dependency(self, tmp_path: Path) -> None:
        dependency_root = tmp_path / "dependency"
        dependency = PackageInfo(
            dependency_root,
            PackageManifest("helpers", semver.Version.parse("1.0.0")),
        )
        dependency.module_root.mkdir(parents=True)
        (dependency.module_root / "api.agl").write_text("let value = 1\n")
        consumer = _custom_package(tmp_path / "consumer")
        (consumer.module_root / "main.agl").write_text("import helpers/api\n")

        with pytest.raises(DisciplineError):
            validate_package(consumer, dependency_packages=(dependency,))

    def test_accepts_import_from_declared_dependency(self, tmp_path: Path) -> None:
        dependency_root = tmp_path / "dependency"
        dependency = PackageInfo(
            dependency_root,
            PackageManifest("helpers", semver.Version.parse("1.0.0")),
        )
        dependency.module_root.mkdir(parents=True)
        (dependency.module_root / "api.agl").write_text("let value = 1\n")
        consumer_root = tmp_path / "consumer"
        consumer = PackageInfo(
            consumer_root,
            PackageManifest(
                "consumer",
                semver.Version.parse("1.0.0"),
                dependencies={
                    "helpers": DependencySpec(semver.Version.parse("1.0.0")),
                },
            ),
        )
        consumer.module_root.mkdir(parents=True)
        (consumer.module_root / "main.agl").write_text("import helpers/api\n")

        validate_package(consumer, dependency_packages=(dependency,))

    def test_rejects_missing_import_used_by_registered_command(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path, command_path="start")
        (package.module_root / "main.agl").write_text(
            "import custom/missing\nprogram def main() -> unit = ()\n"
        )

        with pytest.raises(DisciplineError):
            validate_package(package)

    def test_rejects_archive_import_of_missing_package_module(self) -> None:
        modules = {"custom/main.agl": "import custom/missing\n"}
        manifest = PackageManifest("custom", semver.Version.parse("1.0.0"))

        with pytest.raises(DisciplineError):
            validate_archive_package(
                manifest,
                archive_paths=modules,
                read_module=modules.__getitem__,
            )

    def test_defers_archive_import_validation_until_dependencies_are_available(self) -> None:
        modules = {"custom/main.agl": "import helpers/api\n"}
        manifest = PackageManifest(
            "custom",
            semver.Version.parse("1.0.0"),
            dependencies={
                "helpers": DependencySpec(semver.Version.parse("1.0.0")),
            },
        )

        validate_archive_package(
            manifest,
            archive_paths=modules,
            read_module=modules.__getitem__,
        )

    def test_archive_import_validation_accepts_a_bundled_extern_companion(self) -> None:
        files = {
            "custom/main.agl": "extern def execute() -> unit\n",
            "custom/main.py": "def execute():\n    return None\n",
        }
        manifest = PackageManifest("custom", semver.Version.parse("1.0.0"))

        validate_archive_package(
            manifest,
            archive_paths=files,
            read_module=files.__getitem__,
        )

    def test_archive_materialization_oserror_is_a_discipline_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        modules = {"custom/main.agl": "let value = 1\n"}
        manifest = PackageManifest("custom", semver.Version.parse("1.0.0"))

        def fail_to_create_temporary_directory(*, prefix: str) -> object:
            raise OSError(f"temporary directory unavailable for {prefix}")

        monkeypatch.setattr(
            "agm.packages.discipline.TemporaryDirectory",
            fail_to_create_temporary_directory,
        )

        with pytest.raises(DisciplineError) as raised:
            validate_archive_package(
                manifest,
                archive_paths=modules,
                read_module=modules.__getitem__,
            )

        assert isinstance(raised.value.__cause__, OSError)

    @pytest.mark.parametrize("fixture", ("malformed_command", "reserved_command"))
    def test_rejects_malformed_or_reserved_command_path(self, fixture: str) -> None:
        with pytest.raises(DisciplineError):
            validate_package(_package(fixture))

    @pytest.mark.parametrize("alias", ("wsp", "wt"))
    def test_rejects_command_path_starting_with_a_builtin_alias(
        self, tmp_path: Path, alias: str
    ) -> None:
        package = _custom_package(tmp_path, command_path=f"{alias} launch")
        (package.module_root / "main.agl").write_text("program def main() -> unit = ()\n")

        with pytest.raises(DisciplineError):
            validate_package(package)

    @pytest.mark.parametrize("command", ("cp", "copy"))
    def test_accepts_command_path_matching_a_config_subcommand(
        self, tmp_path: Path, command: str
    ) -> None:
        package = _custom_package(tmp_path, command_path=command)
        (package.module_root / "main.agl").write_text("program def main() -> unit = ()\n")

        validate_package(package)

    @pytest.mark.parametrize("command_path", ("--launch", "tools --help", "tools --dry-run"))
    def test_rejects_command_path_containing_an_option(
        self, tmp_path: Path, command_path: str
    ) -> None:
        package = _custom_package(tmp_path, command_path=command_path)
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

    @pytest.mark.parametrize(
        "declaration",
        (
            "program def main[T]() -> unit = ()",
            "program def main(value: int) -> unit = ()",
            "program def main() -> int = 1",
            "program def main() = ()",
        ),
    )
    def test_rejects_registered_program_with_invalid_entry_signature(
        self, tmp_path: Path, declaration: str
    ) -> None:
        package = _custom_package(tmp_path, command_path="start")
        (package.module_root / "main.agl").write_text(f"{declaration}\n")

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

    def test_rejects_symlinked_module_file_escaping_module_tree(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path, command_path="start")
        outside = tmp_path / "outside.agl"
        outside.write_text("program def main() -> unit = ()\n")
        (package.module_root / "main.agl").symlink_to(outside)

        with pytest.raises(DisciplineError, match="escapes"):
            validate_package(package)

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

    def test_accepts_nonexpanding_resource_reexport_cycle(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "a.agl").write_text(
            "export std/core using resource\nexport custom/b\n"
        )
        (package.module_root / "b.agl").write_text("export custom/a\n")

        validate_package(package)

    def test_rejects_cyclic_scoped_resource_reexports_without_hanging(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "a.agl").write_text(
            "export std/core using resource\nscope Loop\nexport custom/b\nend Loop\n"
        )
        (package.module_root / "b.agl").write_text("scope Loop\nexport custom/a\nend Loop\n")

        with (
            fail_if_slow("resource re-export resolution did not terminate"),
            pytest.raises(DisciplineError),
        ):
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

    def test_rejects_missing_resource_through_an_open_alias(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "main.agl").write_text(
            "open Assets using asset as load\n"
            "scope Assets\n"
            "import std/core using resource as asset\n"
            "end Assets\n"
            'let prompt = load("prompts/missing.md")\n'
        )

        with pytest.raises(DisciplineError):
            validate_package(package)

    @pytest.mark.parametrize(
        "open_declaration",
        (
            "open Assets",
            "open Assets hiding directories",
            "open Assets using absent, asset as load",
        ),
    )
    def test_resource_open_selection_modes_are_tracked(
        self, tmp_path: Path, open_declaration: str
    ) -> None:
        package = _custom_package(tmp_path)
        called = (
            "asset" if open_declaration != "open Assets using absent, asset as load" else "load"
        )
        (package.module_root / "main.agl").write_text(
            f"{open_declaration}\n"
            "scope Assets\n"
            "import std/core using resource as asset, resource-dir as directories\n"
            "end Assets\n"
            f'let prompt = {called}("prompts/missing.md")\n'
        )

        with pytest.raises(DisciplineError):
            validate_package(package)

    def test_resource_open_without_resource_members_is_ignored(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "main.agl").write_text(
            "open Empty\nscope Empty\ndef value() -> unit = ()\nend Empty\n"
        )

        validate_package(package)

    @pytest.mark.parametrize(
        ("declaration", "constructor"),
        (
            ("record resource(value: text)", "resource"),
            ("enum Value | resource(value: text)", "resource"),
            ("exception resource(value: text)", "resource"),
            ("record resource-dir(value: text)", "resource-dir"),
            ("enum Value | resource-dir(value: text)", "resource-dir"),
            ("exception resource-dir(value: text)", "resource-dir"),
        ),
    )
    def test_accepts_constructor_named_like_resource_builtin(
        self, tmp_path: Path, declaration: str, constructor: str
    ) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "main.agl").write_text(
            f'{declaration}\nlet value = {constructor}(value = "not/a/resource")\n'
        )

        validate_package(package)

    def test_accepts_resource_alias_shadowed_by_a_function_parameter(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "main.agl").write_text(
            "import std/core using resource as asset\n"
            'def use(asset: (text) -> text) -> text = asset("not/a/resource")\n'
        )

        validate_package(package)

    @pytest.mark.parametrize("binding", ("let", "var"))
    def test_accepts_resource_alias_shadowed_by_a_local_binding(
        self, tmp_path: Path, binding: str
    ) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "main.agl").write_text(
            "import std/core using resource as asset\n"
            "program def main() -> text =\n"
            f"  {binding} asset = fn(path: text) => path\n"
            '  asset("not/a/resource")\n'
        )

        validate_package(package)

    def test_accepts_resource_alias_shadowed_inside_a_nested_block(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "main.agl").write_text(
            "import std/core using resource as asset\n"
            "def load(path: text) -> text = path\n"
            "program def main() -> unit =\n"
            "  if true =>\n"
            "    let asset = load\n"
            '    print (asset("not/a/resource"))\n'
            "  else => ()\n"
        )

        validate_package(package)

    def test_rejects_function_declared_with_a_resource_builtin_name(self, tmp_path: Path) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "main.agl").write_text("def resource(path: text) -> text = path\n")

        with pytest.raises(DisciplineError):
            validate_package(package)

    def test_accepts_scoped_enum_constructor_shadowing_resource_import(
        self, tmp_path: Path
    ) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "main.agl").write_text(
            "scope Types\n"
            "scope Value\n"
            "import std/core using resource\n"
            "end Value\n"
            "enum Value | resource(value: text)\n"
            "end Types\n"
            'let value = Types::Value::resource(value = "not/a/resource")\n'
        )

        validate_package(package)

    def test_rejects_missing_resource_reexported_by_dependency(self, tmp_path: Path) -> None:
        dependency_root = tmp_path / "dependency"
        (dependency_root / "helpers").mkdir(parents=True)
        dependency = PackageInfo(
            dependency_root,
            PackageManifest("helpers", semver.Version.parse("1.0.0")),
        )
        (dependency.module_root / "assets.agl").write_text(
            "export std/core using resource as asset\n"
        )
        consumer = _custom_package(
            tmp_path / "consumer",
            dependencies={"helpers": DependencySpec(semver.Version.parse("1.0.0"))},
        )
        (consumer.module_root / "main.agl").write_text(
            'import helpers/assets using asset as load\nlet prompt = load("prompts/missing.md")\n'
        )

        with pytest.raises(DisciplineError, match="missing.md"):
            validate_package(consumer, dependency_packages=(dependency,))

    def test_rejects_unparseable_dependency_module_reached_by_an_import(
        self, tmp_path: Path
    ) -> None:
        dependency_root = tmp_path / "dependency"
        (dependency_root / "helpers").mkdir(parents=True)
        dependency = PackageInfo(
            dependency_root,
            PackageManifest("helpers", semver.Version.parse("1.0.0")),
        )
        (dependency.module_root / "assets.agl").write_text("export ???\n")
        consumer = _custom_package(
            tmp_path / "consumer",
            dependencies={"helpers": DependencySpec(semver.Version.parse("1.0.0"))},
        )
        (consumer.module_root / "main.agl").write_text("import helpers/assets\n")

        with pytest.raises(DisciplineError):
            validate_package(consumer, dependency_packages=(dependency,))

    def test_rejects_missing_resource_through_a_scoped_alias_reexport_chain(
        self, tmp_path: Path
    ) -> None:
        package = _custom_package(tmp_path)
        (package.module_root / "resources.agl").write_text(
            "scope Assets\nexport std/core using resource as asset\nend Assets\n"
        )
        (package.module_root / "facade.agl").write_text(
            "export custom/resources using Assets::asset\n"
        )
        (package.module_root / "main.agl").write_text(
            "import custom/facade using Assets::asset as load\n"
            'let prompt = load("prompts/missing.md")\n'
        )

        with pytest.raises(DisciplineError):
            validate_package(package)

    def test_rejects_invalid_utf8_referenced_source_fixture(self) -> None:
        with pytest.raises(DisciplineError):
            validate_package(_package("invalid_utf8_source"))
