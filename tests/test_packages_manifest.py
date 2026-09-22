"""Package manifest parsing and validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import semver

from agm.agl.keywords import KEYWORDS
from agm.packages.archive import extract_archive, write_archive
from agm.packages.distribution import normalized_manifest
from agm.packages.layout import MODULE_TREE_DIRNAME
from agm.packages.manifest import (
    CommandSpec,
    ManifestError,
    UnknownField,
    distribution_manifest,
    expanded_commands,
    load_manifest,
    load_manifest_text,
    validate_command_set,
)

FIXTURES = Path(__file__).parent / "agl" / "packages"
URL = "https://example.test/tools.agmpkg"


def _write_manifest(tmp_path: Path, text: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "package.toml"
    manifest_path.write_text(text, encoding="utf-8")
    return manifest_path


class TestPackageManifest:
    def test_distribution_manifest_strips_path_sources_without_mutating_source(
        self, tmp_path: Path
    ) -> None:
        manifest = load_manifest(
            _write_manifest(
                tmp_path,
                """[package]
name = "alpha"
version = "1.0.0"

[dependencies]
bravo = { version = "2", path = "../bravo" }
charlie = { version = "3", url = "https://example.test/charlie.agmpkg", hash = "sha256:"""
                + "a" * 64
                + '" }\n',
            )
        )

        distribution = distribution_manifest(manifest)

        assert manifest.dependencies["bravo"].path == "../bravo"
        assert distribution.dependencies["bravo"].path is None
        assert distribution.dependencies["bravo"].version == manifest.dependencies["bravo"].version
        assert distribution.dependencies["charlie"] == manifest.dependencies["charlie"]
        assert distribution is not manifest

    def test_loads_complete_manifest(self) -> None:
        manifest = load_manifest(FIXTURES / "valid" / "package.toml")

        assert manifest.name == "review_tools"
        assert manifest.version == semver.Version.parse("0.2.0")
        assert manifest.description == "Review workflow commands"
        assert manifest.authors == ("AGM",)
        assert manifest.dependencies["std"].version == semver.Version.parse("0.1.0")
        assert manifest.dependencies["judge"].path == "../judge"
        assert manifest.dependencies["tools"].url == "https://example.test/tools.agmpkg"
        assert (
            manifest.dependencies["tools"].hash
            == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
        assert manifest.commands["review-loop"].program == "review_tools/review::main"
        assert manifest.commands["review-loop lint"].doc == "Lint review configurations"

    @pytest.mark.parametrize("name", ("review_tools", "review-tools", "Review2", "_internal"))
    def test_accepts_module_segment_package_names(self, tmp_path: Path, name: str) -> None:
        manifest = load_manifest(
            _write_manifest(tmp_path, f'[package]\nname = "{name}"\nversion = "1.2.3"\n')
        )

        assert manifest.name == name

    @pytest.mark.parametrize("name", ("-review", "1review", "review/tools"))
    def test_rejects_invalid_package_names(self, tmp_path: Path, name: str) -> None:
        path = _write_manifest(tmp_path, f'[package]\nname = "{name}"\nversion = "1.2.3"\n')

        with pytest.raises(ManifestError):
            load_manifest(path)

    @pytest.mark.parametrize("name", sorted(KEYWORDS))
    def test_rejects_reserved_keyword_package_names(self, tmp_path: Path, name: str) -> None:
        path = _write_manifest(tmp_path, f'[package]\nname = "{name}"\nversion = "1.2.3"\n')

        with pytest.raises(ManifestError):
            load_manifest(path)

    @pytest.mark.parametrize("name", sorted(KEYWORDS))
    def test_rejects_reserved_keyword_dependency_names(self, tmp_path: Path, name: str) -> None:
        path = _write_manifest(
            tmp_path,
            '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n'
            f'[dependencies]\n"{name}" = "1"\n',
        )

        with pytest.raises(ManifestError):
            load_manifest(path)

    @pytest.mark.parametrize("version", ("1.2.3", "1.2.3-alpha.1", "1.2.3+build.5"))
    def test_accepts_semver_versions(self, tmp_path: Path, version: str) -> None:
        manifest = load_manifest(
            _write_manifest(tmp_path, f'[package]\nname = "review_tools"\nversion = "{version}"\n')
        )

        assert manifest.version == semver.Version.parse(version)

    @pytest.mark.parametrize("version", ("1.2", "v1.2.3", "1.2.3.4", "01.2.3"))
    def test_rejects_non_semver_versions(self, tmp_path: Path, version: str) -> None:
        path = _write_manifest(
            tmp_path, f'[package]\nname = "review_tools"\nversion = "{version}"\n'
        )

        with pytest.raises(ManifestError):
            load_manifest(path)

    def test_preserves_semver_prerelease_precedence(self, tmp_path: Path) -> None:
        alpha = load_manifest(
            _write_manifest(
                tmp_path / "alpha", '[package]\nname = "review_tools"\nversion = "1.0.0-alpha"\n'
            )
        )
        release = load_manifest(
            _write_manifest(
                tmp_path / "release", '[package]\nname = "review_tools"\nversion = "1.0.0"\n'
            )
        )

        assert alpha.version < release.version

    @pytest.mark.parametrize(
        ("entry", "expected_version"),
        (
            ('std = "0.4"', "0.4.0"),
            ('judge = { version = "0.1", path = "../judge" }', "0.1.0"),
            (
                f'tools = {{ version = "1.1", url = "{URL}", hash = "sha256:{"a" * 64}" }}',
                "1.1.0",
            ),
        ),
    )
    def test_accepts_dependency_source_shapes(
        self, tmp_path: Path, entry: str, expected_version: str
    ) -> None:
        manifest = load_manifest(
            _write_manifest(
                tmp_path,
                '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n[dependencies]\n'
                + entry
                + "\n",
            )
        )

        assert len(manifest.dependencies) == 1
        assert next(iter(manifest.dependencies.values())).version == semver.Version.parse(
            expected_version
        )

    @pytest.mark.parametrize(
        ("requirement", "expected"),
        (
            ("1", "1.0.0"),
            ("1.2", "1.2.0"),
            ("1.2.3", "1.2.3"),
            ("1.2-alpha", "1.2.0-alpha"),
            ("1.2.3-alpha.1+build.5", "1.2.3-alpha.1+build.5"),
        ),
    )
    def test_normalizes_minimum_version_requirements(
        self, tmp_path: Path, requirement: str, expected: str
    ) -> None:
        manifest = load_manifest(
            _write_manifest(
                tmp_path,
                '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n'
                f'[dependencies]\nstd = "{requirement}"\n',
            )
        )

        assert manifest.dependencies["std"].version == semver.Version.parse(expected)

    def test_preserves_prerelease_precedence_for_minimum_version_requirements(
        self, tmp_path: Path
    ) -> None:
        alpha = load_manifest(
            _write_manifest(
                tmp_path / "alpha",
                '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n'
                '[dependencies]\nstd = "1.0.0-alpha"\n',
            )
        )
        release = load_manifest(
            _write_manifest(
                tmp_path / "release",
                '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n'
                '[dependencies]\nstd = "1.0"\n',
            )
        )
        build = load_manifest(
            _write_manifest(
                tmp_path / "build",
                '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n'
                '[dependencies]\nstd = "1.0.0+build.1"\n',
            )
        )

        assert alpha.dependencies["std"].version < release.dependencies["std"].version
        assert build.dependencies["std"].version == release.dependencies["std"].version

    @pytest.mark.parametrize("requirement", ("01", "1.02", "1.2.03", "1.2.3.4"))
    def test_rejects_malformed_minimum_version_requirements(
        self, tmp_path: Path, requirement: str
    ) -> None:
        path = _write_manifest(
            tmp_path,
            '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n'
            f'[dependencies]\nstd = "{requirement}"\n',
        )

        with pytest.raises(ManifestError):
            load_manifest(path)

    @pytest.mark.parametrize("dependency_path", ("../judge", "sources/judge", r"..\judge"))
    def test_accepts_relative_dependency_paths_for_all_platforms(
        self, tmp_path: Path, dependency_path: str
    ) -> None:
        manifest = load_manifest(
            _write_manifest(
                tmp_path,
                '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n[dependencies]\n'
                f"judge = {{ version = \"1.1.0\", path = '{dependency_path}' }}\n",
            )
        )

        assert manifest.dependencies["judge"].path == dependency_path

    @pytest.mark.parametrize(
        "dependency_path",
        (
            "/judge",
            r"\judge",
            "C:/judge",
            r"C:\judge",
            "C:judge",
            "//server/share/judge",
            r"\\server\share\judge",
        ),
    )
    def test_rejects_absolute_dependency_paths_for_all_platforms(
        self, tmp_path: Path, dependency_path: str
    ) -> None:
        path = _write_manifest(
            tmp_path,
            '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n[dependencies]\n'
            f"judge = {{ version = \"1.1.0\", path = '{dependency_path}' }}\n",
        )

        with pytest.raises(ManifestError):
            load_manifest(path)

    @pytest.mark.parametrize(
        "entry",
        (
            'judge = { path = "../judge" }',
            'judge = { version = "1.1.0", path = "/judge" }',
            f'tools = {{ version = "1.1.0", url = "{URL}" }}',
            f'tools = {{ version = "1.1.0", url = "{URL}", hash = "bogus" }}',
            f'tools = {{ version = "1.1.0", url = "{URL}", hash = "sha256:abc" }}',
            'tools = { version = "1.1.0", hash = "sha256:abc" }',
            (
                f'tools = {{ version = "1.1.0", path = "../tools", url = "{URL}", '
                'hash = "sha256:abc" }'
            ),
        ),
    )
    def test_rejects_invalid_dependency_source_shapes(self, tmp_path: Path, entry: str) -> None:
        path = _write_manifest(
            tmp_path,
            '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n[dependencies]\n'
            + entry
            + "\n",
        )

        with pytest.raises(ManifestError):
            load_manifest(path)

    @pytest.mark.parametrize(
        "entry",
        ("std = 1",),
    )
    def test_rejects_invalid_dependency_entries(self, tmp_path: Path, entry: str) -> None:
        path = _write_manifest(
            tmp_path,
            '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n[dependencies]\n'
            + entry
            + "\n",
        )

        with pytest.raises(ManifestError):
            load_manifest(path)

    def test_nested_command_tables_match_space_separated_paths(self) -> None:
        package = '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n'
        nested = load_manifest_text(
            package
            + """[commands.devel]
doc = "Development workflows"

[commands.devel.review]
program = "review_tools/main::review"

[commands.devel.quality.lint]
program = "review_tools/main::lint"
"""
        )
        space_separated = load_manifest_text(
            package
            + """[commands.devel]
doc = "Development workflows"

[commands."devel review"]
program = "review_tools/main::review"

[commands."devel quality lint"]
program = "review_tools/main::lint"
"""
        )

        assert nested.commands == space_separated.commands

    def test_nested_command_tables_allow_metadata_names_as_path_components(self) -> None:
        manifest = load_manifest_text(
            """[package]
name = "review_tools"
version = "1.2.3"

[commands.devel.program]
program = "review_tools/main::review"
"""
        )

        assert manifest.commands["devel program"].program == "review_tools/main::review"

    def test_rejects_command_paths_defined_by_both_nested_and_quoted_tables(self) -> None:
        manifest = """[package]
name = "review_tools"
version = "1.2.3"

[commands."devel review"]
program = "review_tools/main::review"

[commands.devel.review]
program = "review_tools/main::review"
"""

        with pytest.raises(ManifestError):
            load_manifest_text(manifest)

    @pytest.mark.parametrize(
        "commands",
        (
            "[commands]\nrun = 'review_tools/main::main'\n",
            "[commands]\nrun = { doc = 'missing program' }\n",
        ),
    )
    def test_rejects_invalid_command_entries(self, tmp_path: Path, commands: str) -> None:
        path = _write_manifest(
            tmp_path, '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n' + commands
        )

        with pytest.raises(ManifestError):
            load_manifest(path)

    @pytest.mark.parametrize(
        "manifest",
        (
            'dependencies = "std"\n\n[package]\nname = "review_tools"\nversion = "1.2.3"\n',
            'commands = []\n\n[package]\nname = "review_tools"\nversion = "1.2.3"\n',
        ),
    )
    def test_rejects_unknown_keys_and_malformed_optional_tables(
        self, tmp_path: Path, manifest: str
    ) -> None:
        with pytest.raises(ManifestError):
            load_manifest(_write_manifest(tmp_path, manifest))

    @pytest.mark.parametrize(
        ("manifest", "expected"),
        (
            (
                '[package]\nname = "review_tools"\nversion = "1.2.3"\nrelease = "beta"\n',
                (UnknownField("package", "release"),),
            ),
            (
                'unexpected = true\n\n[package]\nname = "review_tools"\nversion = "1.2.3"\n',
                (UnknownField("manifest", "unexpected"),),
            ),
            (
                '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n'
                '[dependencies]\nstd = { version = "1.0.0", source = "elsewhere" }\n',
                (UnknownField("dependency 'std'", "source"),),
            ),
            (
                '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n'
                "[commands]\nrun = { program = 'review_tools/main::main', extra = 'x' }\n",
                (UnknownField("command 'run'", "extra"),),
            ),
        ),
    )
    def test_records_fields_the_schema_does_not_define(
        self, manifest: str, expected: tuple[UnknownField, ...]
    ) -> None:
        """Loading keeps reading; validation is what rejects them."""
        loaded = load_manifest_text(manifest)

        assert loaded.unknown_fields == expected
        assert loaded.name == "review_tools"

    def test_an_unknown_field_does_not_displace_the_ones_beside_it(self) -> None:
        loaded = load_manifest_text(
            '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n'
            "[commands]\nrun = { program = 'review_tools/main::run', doc = 'Run', extra = 'x' }\n"
        )

        assert loaded.commands["run"] == CommandSpec("review_tools/main::run", "Run")

    def test_rejects_invalid_utf8_manifest(self, tmp_path: Path) -> None:
        path = tmp_path / "package.toml"
        path.write_bytes(b"\xff")

        with pytest.raises(ManifestError):
            load_manifest(path)

    def test_rejects_invalid_metadata_and_toml(self, tmp_path: Path) -> None:
        invalid_metadata = _write_manifest(
            tmp_path / "metadata",
            '[package]\nname = "review_tools"\nversion = "1.2.3"\nauthors = [1]\n',
        )
        invalid_toml = _write_manifest(tmp_path / "toml", "[package\n")

        with pytest.raises(ManifestError):
            load_manifest(invalid_metadata)
        with pytest.raises(ManifestError):
            load_manifest(invalid_toml)

    def test_rejects_missing_manifest_file(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError):
            load_manifest(tmp_path / "package.toml")

    @pytest.mark.parametrize(
        "manifest",
        (
            "",
            "[package]\nname = 'review_tools'\n",
            "[package]\nversion = '1.2.3'\n",
            "[package]\nname = 1\nversion = '1.2.3'\n",
        ),
    )
    def test_rejects_missing_or_invalid_required_fields(
        self, tmp_path: Path, manifest: str
    ) -> None:
        with pytest.raises(ManifestError):
            load_manifest(_write_manifest(tmp_path, manifest))

    _LONELY_GROUP = (
        '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n'
        '[commands.devel]\ndoc = "Development workflows"\n'
    )
    _ALIAS_TO_UNKNOWN = (
        '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n[aliases]\nrev = "devel review"\n'
    )

    def test_a_complete_manifest_with_a_lonely_group_is_still_rejected_at_load(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ManifestError, match="devel"):
            load_manifest(_write_manifest(tmp_path, self._LONELY_GROUP))

    def test_a_complete_manifest_with_an_alias_to_nothing_is_still_rejected_at_load(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ManifestError, match="rev"):
            load_manifest(_write_manifest(tmp_path, self._ALIAS_TO_UNKNOWN))

    def test_commands_complete_false_defers_the_lonely_group_check(self, tmp_path: Path) -> None:
        manifest = load_manifest(
            _write_manifest(tmp_path, self._LONELY_GROUP), commands_complete=False
        )

        assert manifest.commands["devel"].program is None
        with pytest.raises(ManifestError, match="devel"):
            validate_command_set(manifest)

    def test_commands_complete_false_defers_the_unknown_alias_target_check(
        self, tmp_path: Path
    ) -> None:
        manifest = load_manifest(
            _write_manifest(tmp_path, self._ALIAS_TO_UNKNOWN), commands_complete=False
        )

        assert manifest.aliases["rev"] == "devel review"
        with pytest.raises(ManifestError, match="rev"):
            validate_command_set(manifest)

    def test_a_deferred_manifest_passes_once_a_descendant_is_merged_in(
        self, tmp_path: Path
    ) -> None:
        manifest = load_manifest(
            _write_manifest(tmp_path, self._LONELY_GROUP), commands_complete=False
        )

        merged = replace(
            manifest,
            commands={
                **manifest.commands,
                "devel review": CommandSpec(program="review_tools/main::review"),
            },
        )

        validate_command_set(merged)

    def test_a_deferred_alias_resolves_once_its_target_is_merged_in(self, tmp_path: Path) -> None:
        manifest = load_manifest(
            _write_manifest(tmp_path, self._ALIAS_TO_UNKNOWN), commands_complete=False
        )

        merged = replace(
            manifest,
            commands={"devel review": CommandSpec(program="review_tools/main::review")},
        )

        validate_command_set(merged)
        assert expanded_commands(merged)["rev"].program == "review_tools/main::review"

    def test_commands_complete_false_still_validates_alias_path_syntax_at_load(
        self, tmp_path: Path
    ) -> None:
        manifest = (
            '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n'
            '[aliases]\n"bad  path" = "devel review"\n'
        )

        with pytest.raises(ManifestError):
            load_manifest(_write_manifest(tmp_path, manifest), commands_complete=False)


class TestManifestRejectsLoneSurrogateEscapes:
    """A manifest value must be valid Unicode; tomlkit's 8-digit ``\\U0000Dxxx``
    escape can spell the same lone surrogate its 4-digit ``\\uDxxx`` form is
    already rejected for, so both must fail manifest loading the same way."""

    _MANIFEST = (
        '[package]\nname = "review_tools"\nversion = "1.2.3"\n'
        'description = "' + "\\U0000" + 'D800"\n'
    )

    def test_load_manifest_text_rejects_it(self) -> None:
        with pytest.raises(ManifestError):
            load_manifest_text(self._MANIFEST)

    def test_load_manifest_rejects_it(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError):
            load_manifest(_write_manifest(tmp_path, self._MANIFEST))


class TestPythonDependencies:
    """The ``[python] dependencies`` table: PEP 508 requirements a package's companions import."""

    _HEADER = '[package]\nname = "review_tools"\nversion = "1.2.3"\n\n'

    def _package(self, root: Path, dependencies: str) -> Path:
        (root / MODULE_TREE_DIRNAME).mkdir(parents=True)
        (root / MODULE_TREE_DIRNAME / "main.agl").write_text(
            "program def main() -> unit = ()\n", encoding="utf-8"
        )
        (root / "package.toml").write_text(
            self._HEADER + f"[python]\ndependencies = {dependencies}\n", encoding="utf-8"
        )
        return root

    def test_parses_requirements_verbatim_in_declared_order(self) -> None:
        manifest = load_manifest_text(
            self._HEADER + "[python]\n"
            "dependencies = ['typesafe-sdk >= 0.7, < 1', \"legacy ; python_version < '3'\"]\n"
        )

        assert manifest.python_dependencies == (
            "typesafe-sdk >= 0.7, < 1",
            "legacy ; python_version < '3'",
        )

    def test_absent_table_declares_nothing(self) -> None:
        assert load_manifest_text(self._HEADER).python_dependencies == ()

    @pytest.mark.parametrize(
        "table",
        (
            "[python]\ndependencies = 'requests'\n",
            "[python]\ndependencies = ['']\n",
            "[python]\ndependencies = [1]\n",
            "[python]\ndependencies = ['not a requirement!']\n",
            "[python]\ndependencies = ['requests >= one']\n",
            "[python]\ndependencies = ['requests @ https://example.test/requests.whl']\n",
            "[python]\ndependencies = [\"rich; extra == 'cli'\"]\n",
            "[python]\ndependencies = [\"rich; python_version > '3' and 'cli' == extra\"]\n",
            "[python]\ndependencies = [\"rich; 'cli' in extras\"]\n",
            "[python]\ndependencies = [\"rich; 'dev' in dependency_groups\"]\n",
            "[python]\ndependencies = [\"rich; os_name ~= 'posix'\"]\n",
            "[python]\ndependencies = [\"rich; python_version < '3' and 'cli' in extras\"]\n",
        ),
    )
    def test_rejects_malformed_requirements(self, table: str) -> None:
        with pytest.raises(ManifestError):
            load_manifest_text(self._HEADER + table)

    def test_rejects_a_non_table_python_entry(self) -> None:
        with pytest.raises(ManifestError):
            load_manifest_text("python = 'requests'\n" + self._HEADER)

    def test_records_unknown_python_fields(self) -> None:
        manifest = load_manifest_text(self._HEADER + "[python]\nindex = 'https://example.test'\n")

        assert manifest.unknown_fields == (UnknownField("python", "index"),)
        assert manifest.python_dependencies == ()

    def test_normalized_manifest_round_trips_requirements(self) -> None:
        manifest = load_manifest_text(
            self._HEADER + "[python]\ndependencies = ['b>=1', 'a[extra]<2']\n"
        )

        rendered = normalized_manifest(manifest).decode()

        assert load_manifest_text(rendered) == manifest
        assert normalized_manifest(load_manifest_text(rendered)).decode() == rendered

    def test_requirements_change_the_distribution_hash(self, tmp_path: Path) -> None:
        plain = self._package(tmp_path / "plain", "[]")
        required = self._package(tmp_path / "required", "['requests>=2']")

        plain_hash = write_archive(plain, tmp_path / "plain.agmpkg").package_hash
        required_hash = write_archive(required, tmp_path / "required.agmpkg").package_hash

        assert plain_hash != required_hash

    def test_archive_round_trips_requirements(self, tmp_path: Path) -> None:
        root = self._package(tmp_path / "review_tools", "['requests >= 2', 'rich']")
        archive = tmp_path / "review_tools.agmpkg"
        write_archive(root, archive)

        extracted = extract_archive(archive, tmp_path / "extracted")

        assert extracted.manifest.python_dependencies == ("requests >= 2", "rich")

    def test_archive_round_trips_a_backslash_in_a_marker_value(self, tmp_path: Path) -> None:
        spec = 'demo; platform_release == "a\\\\b"'
        root = self._package(tmp_path / "review_tools", f"['{spec}']")
        archive = tmp_path / "review_tools.agmpkg"
        write_archive(root, archive)

        extracted = extract_archive(archive, tmp_path / "extracted")

        assert extracted.manifest.python_dependencies == (spec,)

    def test_marker_values_merely_spelling_extra_are_accepted(self) -> None:
        spec = "rich; platform_release == 'extra'"
        manifest = load_manifest_text(self._HEADER + f'[python]\ndependencies = ["{spec}"]\n')

        assert manifest.python_dependencies == (spec,)

    def test_accepts_per_environment_variants_of_one_distribution(self) -> None:
        specs = ("tomli>=2; python_version < '3.11'", "Tomli<2; python_version < '3'")
        rendered = ", ".join(f'"{spec}"' for spec in specs)
        manifest = load_manifest_text(self._HEADER + f"[python]\ndependencies = [{rendered}]\n")

        assert manifest.python_dependencies == specs
