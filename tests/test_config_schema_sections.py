"""AGM's schema config sections are neither AgL module routes nor registrable names."""

from __future__ import annotations

from pathlib import Path

import pytest
import semver
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
from agm.config.general import GeneralConfig
from agm.config.qualified_keys import QualifiedConfigKey, resolve_qualified_values
from agm.core.toml import toml_dict
from agm.packages.discipline import DisciplineError, validate_package
from agm.packages.errors import PackageInstallError
from agm.packages.layout import MODULE_TREE_DIRNAME
from agm.packages.manifest import CommandSpec, PackageManifest
from agm.packages.model import PackageInfo
from agm.packages.record import write_record
from tests._package_helpers import install_directory

_SCHEMA_SECTIONS = ("agent", "deps", "modules", "packages", "params")


def _package(tmp_path: Path, name: str, command_path: str) -> PackageInfo:
    root = tmp_path / "package"
    module_root = root / MODULE_TREE_DIRNAME
    module_root.mkdir(parents=True)
    (module_root / "main.agl").write_text("program def main() -> unit = ()\n")
    return PackageInfo(
        root=root,
        manifest=PackageManifest(
            name,
            semver.Version.parse("1.0.0"),
            commands={command_path: CommandSpec(f"{name}/main::main")},
        ),
    )


@pytest.mark.parametrize("section", _SCHEMA_SECTIONS)
def test_a_package_cannot_take_a_schema_section_name(tmp_path: Path, section: str) -> None:
    with pytest.raises(DisciplineError):
        validate_package(_package(tmp_path, section, "launch"))


@pytest.mark.parametrize("section", _SCHEMA_SECTIONS)
def test_a_command_path_cannot_start_at_a_schema_section(tmp_path: Path, section: str) -> None:
    with pytest.raises(DisciplineError):
        validate_package(_package(tmp_path, "custom", f"{section} launch"))


def test_a_command_path_may_contain_a_schema_section_below_its_root(tmp_path: Path) -> None:
    validate_package(_package(tmp_path, "custom", "tools agent"))


@pytest.mark.parametrize(
    ("module_segments", "scope_path"),
    [(("std", "agent"), ()), (("agent",), ("claude",)), (("agent",), ("claude", "opus"))],
)
def test_agent_tables_never_configure_an_agent_module(
    module_segments: tuple[str, ...], scope_path: tuple[str, ...]
) -> None:
    config = GeneralConfig.from_layers(
        [
            toml_dict(
                {
                    "agent": {
                        "effort": "low",
                        "claude": {"effort": "medium", "opus": {"effort": "high"}},
                    }
                }
            )
        ]
    )
    key = QualifiedConfigKey(module_segments, scope_path, "effort")

    assert resolve_qualified_values(config, (key,)) == {}


def _invoke(argv: list[str]) -> Result:
    return CliRunner().invoke(get_command(cli.app), argv, prog_name="agm", catch_exceptions=False)


def _package_source(root: Path, *, root_word: str) -> Path:
    """A package registering ``<root_word> review`` and its alias ``<root_word> rv``."""
    (root / MODULE_TREE_DIRNAME).mkdir(parents=True)
    (root / "package.toml").write_text(
        '[package]\nname = "legacy"\nversion = "1.0.0"\n\n'
        f'[commands]\n"{root_word} review" = {{ program = "legacy/main::main" }}\n\n'
        f'[aliases]\n"{root_word} rv" = "{root_word} review"\n'
    )
    (root / MODULE_TREE_DIRNAME / "main.agl").write_text(
        'program def main() -> unit = print "reviewed"\n'
    )
    return root


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    return home


def test_an_installed_schema_section_registration_keeps_working(tmp_path: Path, home: Path) -> None:
    """A registration an earlier build installed under ``agent`` still dispatches and uninstalls."""
    install_directory(_package_source(tmp_path / "source", root_word="tools"), home=home, env={})
    store = home / ".agm" / "packages" / "legacy" / "1.0.0"
    for path in (store / "package.toml", home / ".agm" / "packages" / "index.toml"):
        path.write_text(path.read_text().replace("tools r", "agent r"))
    write_record(store)

    for command in (["agent", "review"], ["agent", "rv"]):
        result = _invoke(command)
        assert result.exit_code == 0, result.output
        assert result.stdout == "reviewed\n"

    assert _invoke(["pkg", "uninstall", "legacy"]).exit_code == 0
    assert not store.exists()


def test_install_and_check_reject_a_schema_section_registration(tmp_path: Path, home: Path) -> None:
    source = _package_source(tmp_path / "source", root_word="agent")

    with pytest.raises(PackageInstallError):
        install_directory(source, home=home, env={})
    assert _invoke(["pkg", "check", str(source)]).exit_code != 0


def test_check_rejects_a_schema_section_alias(tmp_path: Path, home: Path) -> None:
    source = _package_source(tmp_path / "source", root_word="tools")
    manifest = source / "package.toml"
    manifest.write_text(manifest.read_text().replace('"tools rv"', '"agent rv"'))

    assert _invoke(["pkg", "check", str(source)]).exit_code != 0


def test_check_rejects_a_schema_section_source_command(tmp_path: Path, home: Path) -> None:
    source = tmp_path / "source"
    (source / MODULE_TREE_DIRNAME).mkdir(parents=True)
    (source / "package.toml").write_text('[package]\nname = "legacy"\nversion = "1.0.0"\n')
    (source / MODULE_TREE_DIRNAME / "main.agl").write_text(
        '@command("agent review")\nprogram def main() -> unit = ()\n'
    )

    assert _invoke(["pkg", "check", str(source)]).exit_code != 0
