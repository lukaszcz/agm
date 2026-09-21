"""Source-declared package command discovery and its merge into a manifest."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import semver

from agm.packages.discipline import DisciplineError, validate_package
from agm.packages.layout import MODULE_TREE_DIRNAME
from agm.packages.manifest import CommandSpec, PackageManifest, expanded_commands
from agm.packages.model import PackageInfo
from agm.packages.source_commands import package_with_source_commands
from tests._parse_counts import parse_counts

_skip_if_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="permission tests are meaningless as root (root bypasses file modes)",
)


def _package(
    tmp_path: Path,
    *,
    commands: dict[str, CommandSpec] | None = None,
    aliases: dict[str, str] | None = None,
) -> PackageInfo:
    root = tmp_path / "package"
    (root / MODULE_TREE_DIRNAME).mkdir(parents=True)
    manifest = PackageManifest(
        "tools",
        semver.Version.parse("1.0.0"),
        commands=commands or {},
        aliases=aliases or {},
    )
    return PackageInfo(root, manifest)


def _write(package: PackageInfo, relative: str, text: str) -> None:
    path = package.module_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class TestPackageWithSourceCommands:
    def test_a_package_without_registrations_is_returned_unchanged(self, tmp_path: Path) -> None:
        package = _package(tmp_path)
        _write(package, "main.agl", "program def main() -> unit = ()\n")

        assert package_with_source_commands(package) is package

    def test_a_program_at_module_root_registers_its_command(self, tmp_path: Path) -> None:
        package = _package(tmp_path)
        _write(
            package,
            "review.agl",
            '@command("tools review")\n@doc("Review changes")\nprogram def main() -> unit = ()\n',
        )

        merged = package_with_source_commands(package)

        assert merged.manifest.commands == {
            "tools review": CommandSpec(program="tools/review::main", doc="Review changes")
        }

    def test_a_program_in_a_scope_region_registers_its_qualified_declaration(
        self, tmp_path: Path
    ) -> None:
        package = _package(tmp_path)
        _write(
            package,
            "main.agl",
            "scope Devel\n\n"
            '  @command("devel review")\n'
            "  program def run() -> unit = ()\n"
            "end Devel\n\n"
            "program def main() -> unit = ()\n",
        )

        merged = package_with_source_commands(package)

        assert merged.manifest.commands == {
            "devel review": CommandSpec(program="tools/main::Devel::run")
        }

    def test_documentation_of_an_unregistered_program_registers_nothing(
        self, tmp_path: Path
    ) -> None:
        package = _package(tmp_path)
        _write(package, "main.agl", '@doc("Just a program")\nprogram def main() -> unit = ()\n')

        assert package_with_source_commands(package) is package

    def test_a_malformed_command_path_is_a_discipline_error(self, tmp_path: Path) -> None:
        package = _package(tmp_path)
        _write(package, "main.agl", '@command(" bad")\nprogram def main() -> unit = ()\n')

        with pytest.raises(DisciplineError, match="Command path"):
            package_with_source_commands(package)

    def test_two_programs_claiming_one_path_is_a_discipline_error(self, tmp_path: Path) -> None:
        package = _package(tmp_path)
        _write(package, "a.agl", '@command("tools review")\nprogram def main() -> unit = ()\n')
        _write(package, "b.agl", '@command("tools review")\nprogram def main() -> unit = ()\n')

        with pytest.raises(DisciplineError, match="tools review"):
            package_with_source_commands(package)

    @_skip_if_root
    def test_an_unreadable_module_is_a_discipline_error(self, tmp_path: Path) -> None:
        package = _package(tmp_path)
        module = package.module_root / "main.agl"
        module.write_text("program def main() -> unit = ()\n", encoding="utf-8")
        module.chmod(0o000)
        try:
            with pytest.raises(DisciplineError, match="cannot read"):
                package_with_source_commands(package)
        finally:
            module.chmod(0o644)

    def test_an_undecodable_module_is_a_discipline_error(self, tmp_path: Path) -> None:
        package = _package(tmp_path)
        (package.module_root / "main.agl").write_bytes(b"\xff\xfe\x00")

        with pytest.raises(DisciplineError, match="cannot read"):
            package_with_source_commands(package)

    def test_an_unparsable_module_is_a_discipline_error(self, tmp_path: Path) -> None:
        package = _package(tmp_path)
        _write(package, "main.agl", "program def main( -> unit = ()\n")

        with pytest.raises(DisciplineError, match="cannot parse"):
            package_with_source_commands(package)

    def test_a_program_carrying_config_still_registers_its_command(self, tmp_path: Path) -> None:
        package = _package(tmp_path)
        _write(
            package,
            "review.agl",
            'import std/config\n\n@command("tools review")\n'
            "@config(config::log = true)\nprogram def main() -> unit = ()\n",
        )

        merged = package_with_source_commands(package)

        assert merged.manifest.commands == {
            "tools review": CommandSpec(program="tools/review::main")
        }

    def test_discovered_commands_extend_the_manifest(self, tmp_path: Path) -> None:
        package = _package(tmp_path, commands={"launch": CommandSpec(program="tools/other::main")})
        _write(package, "other.agl", "program def main() -> unit = ()\n")
        _write(
            package,
            "review.agl",
            '@command("tools review")\nprogram def main() -> unit = ()\n',
        )

        merged = package_with_source_commands(package)

        assert merged.manifest.commands == {
            "launch": CommandSpec(program="tools/other::main"),
            "tools review": CommandSpec(program="tools/review::main"),
        }
        assert merged.root == package.root

    def test_a_manifest_command_conflicting_with_a_different_program_is_a_discipline_error(
        self, tmp_path: Path
    ) -> None:
        package = _package(
            tmp_path, commands={"tools review": CommandSpec(program="tools/other::main")}
        )
        _write(
            package,
            "review.agl",
            '@command("tools review")\nprogram def main() -> unit = ()\n',
        )

        with pytest.raises(DisciplineError, match="tools review"):
            package_with_source_commands(package)

    def test_a_manifest_command_naming_a_different_program_is_a_discipline_error(
        self, tmp_path: Path
    ) -> None:
        package = _package(
            tmp_path,
            commands={"tools review": CommandSpec(program="tools/other::main")},
        )
        _write(package, "other.agl", "program def main() -> unit = ()\n")
        _write(
            package,
            "review.agl",
            '@command("tools review")\nprogram def main() -> unit = ()\n',
        )

        with pytest.raises(DisciplineError, match="tools review"):
            package_with_source_commands(package)

    def test_a_command_the_manifest_declares_takes_its_program_documentation(
        self, tmp_path: Path
    ) -> None:
        package = _package(
            tmp_path, commands={"tools review": CommandSpec(program="tools/review::main")}
        )
        _write(
            package,
            "review.agl",
            '@doc("Review changes")\nprogram def main() -> unit = ()\n',
        )

        merged = package_with_source_commands(package)

        assert merged.manifest.commands["tools review"].doc == "Review changes"

    def test_a_path_already_baked_with_an_identical_registration_passes_through(
        self, tmp_path: Path
    ) -> None:
        """An already-baked store or archive manifest re-scans its own source without conflict.

        This is what an archive round trip or a reinstall of a store tree
        does: the manifest already records exactly the registration the
        source declares, so the merge must be a no-op rather than a
        manifest-vs-program conflict.
        """
        package = _package(
            tmp_path,
            commands={
                "tools review": CommandSpec(program="tools/review::main", doc="Review changes")
            },
        )
        _write(
            package,
            "review.agl",
            '@command("tools review")\n@doc("Review changes")\nprogram def main() -> unit = ()\n',
        )

        merged = package_with_source_commands(package)

        assert merged.manifest == package.manifest

    def test_an_alias_colliding_with_a_discovered_command_path_is_a_discipline_error(
        self, tmp_path: Path
    ) -> None:
        package = _package(
            tmp_path,
            commands={"build": CommandSpec(program="tools/b::main")},
            aliases={"shipit": "build"},
        )
        _write(package, "b.agl", "program def main() -> unit = ()\n")
        _write(package, "ship.agl", '@command("shipit")\nprogram def main() -> unit = ()\n')

        with pytest.raises(DisciplineError, match="shipit"):
            package_with_source_commands(package)

    def test_a_broken_command_set_is_validated_even_without_registrations(
        self, tmp_path: Path
    ) -> None:
        """A lenient load defers command-set validation whether or not a program registers.

        No ``@command`` program is discovered here, so the merge is a no-op —
        but the merge must still validate the manifest it was handed rather
        than skip straight past an early return.
        """
        package = _package(tmp_path, commands={"devel": CommandSpec(doc="Development workflows")})
        _write(package, "main.agl", "program def main() -> unit = ()\n")

        with pytest.raises(DisciplineError, match="devel"):
            package_with_source_commands(package)

    def test_a_manifest_group_satisfied_only_by_a_discovered_descendant_is_accepted(
        self, tmp_path: Path
    ) -> None:
        package = _package(tmp_path, commands={"devel": CommandSpec(doc="Development workflows")})
        _write(package, "main.agl", '@command("devel review")\nprogram def review() -> unit = ()\n')

        merged = package_with_source_commands(package)

        assert merged.manifest.commands["devel"].doc == "Development workflows"
        assert merged.manifest.commands["devel review"].program == "tools/main::review"

    def test_an_alias_targeting_a_discovered_command_path_resolves(self, tmp_path: Path) -> None:
        package = _package(
            tmp_path,
            commands={"devel": CommandSpec(doc="Development workflows")},
            aliases={"rev": "devel review"},
        )
        _write(package, "main.agl", '@command("devel review")\nprogram def review() -> unit = ()\n')

        merged = package_with_source_commands(package)

        assert expanded_commands(merged.manifest)["rev"].program == "tools/main::review"


def test_discovery_and_validation_parse_each_module_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discovery and the module graph share one parse of every module.

    Discovery reads the same parsed-module cache the loader fills, so a
    package validated after a command scan never re-parses what the scan
    already read. Counting the cache's builder is the only way to observe
    that the two stages share a parse rather than repeating it.
    """
    package = _package(tmp_path)
    (package.root / MODULE_TREE_DIRNAME / "main.agl").write_text(
        '@command("launch")\nprogram def main() -> unit = ()\n', encoding="utf-8"
    )

    with parse_counts(monkeypatch) as counts:
        validate_package(package_with_source_commands(package))

    module = str((package.root / MODULE_TREE_DIRNAME / "main.agl").resolve())
    assert counts[module] == 1


class TestConstantsFoldedIntoRegistrations:
    """A registration's text is folded from the module's own constants.

    Discovery still reads the AST alone, so an editable package's command
    table carries the same prose a compiled run would show.
    """

    def test_a_command_path_and_prose_interpolate_module_constants(self, tmp_path: Path) -> None:
        package = _package(tmp_path)
        _write(
            package,
            "review.agl",
            'let group = "tools"\n'
            "let rounds = 3\n"
            '@command("%{group} review")\n'
            '@doc("Review changes in %{rounds} rounds.")\n'
            "program def main() -> unit = ()\n",
        )

        merged = package_with_source_commands(package)

        assert merged.manifest.commands == {
            "tools review": CommandSpec(
                program="tools/review::main", doc="Review changes in 3 rounds."
            )
        }

    def test_prose_may_name_a_constant_declared_later_in_the_module(self, tmp_path: Path) -> None:
        package = _package(tmp_path)
        _write(
            package,
            "review.agl",
            '@command("tools review")\n'
            "@doc(prose)\n"
            "program def main() -> unit = ()\n"
            '\nlet prose = "Review the working tree."\n',
        )

        merged = package_with_source_commands(package)

        assert merged.manifest.commands["tools review"].doc == "Review the working tree."

    def test_prose_naming_another_module_is_rejected(self, tmp_path: Path) -> None:
        package = _package(tmp_path)
        _write(package, "shared.agl", 'let prose = "Shared."\n')
        _write(
            package,
            "review.agl",
            "import shared\n"
            '@command("tools review")\n'
            "@doc(shared::prose)\n"
            "program def main() -> unit = ()\n",
        )

        with pytest.raises(DisciplineError, match="constant"):
            package_with_source_commands(package)
