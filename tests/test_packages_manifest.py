"""Package manifest parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import semver

from agm.agl.keywords import KEYWORDS
from agm.packages.manifest import ManifestError, distribution_manifest, load_manifest

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
        assert manifest.dependencies["std"].version == semver.Version.parse("0.2.0")
        assert manifest.dependencies["judge"].path == "../judge"
        assert manifest.dependencies["tools"].url == "https://example.test/tools.agmpkg"
        assert (
            manifest.dependencies["tools"].hash
            == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
        assert manifest.commands["review-loop"].program == "review_tools/review::main"
        assert manifest.commands["review-loop lint"].description == "Lint review configurations"

    @pytest.mark.parametrize("name", ("review_tools", "Review2", "_internal"))
    def test_accepts_module_segment_package_names(self, tmp_path: Path, name: str) -> None:
        manifest = load_manifest(
            _write_manifest(tmp_path, f'[package]\nname = "{name}"\nversion = "1.2.3"\n')
        )

        assert manifest.name == name

    @pytest.mark.parametrize("name", ("review-tools", "1review", "review/tools"))
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
        (
            "std = 1",
            'std = { version = "1.0.0", source = "unsupported" }',
        ),
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

    @pytest.mark.parametrize(
        "commands",
        (
            "[commands]\nrun = 'review_tools/main::main'\n",
            "[commands]\nrun = { description = 'missing program' }\n",
            "[commands]\nrun = { program = 'review_tools/main::main', extra = 'invalid' }\n",
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
            '[package]\nname = "review_tools"\nversion = "1.2.3"\nextra = true\n',
            'unexpected = true\n\n[package]\nname = "review_tools"\nversion = "1.2.3"\n',
            'dependencies = "std"\n\n[package]\nname = "review_tools"\nversion = "1.2.3"\n',
            'commands = []\n\n[package]\nname = "review_tools"\nversion = "1.2.3"\n',
        ),
    )
    def test_rejects_unknown_keys_and_malformed_optional_tables(
        self, tmp_path: Path, manifest: str
    ) -> None:
        with pytest.raises(ManifestError):
            load_manifest(_write_manifest(tmp_path, manifest))

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
