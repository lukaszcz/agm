"""Package command help, group discovery, and alias configuration workflows."""

from pathlib import Path

import pytest
from click.testing import CliRunner
from typer.main import get_command

from agm.cli import app
from agm.packages.install import install_directory
from agm.packages.manifest import ManifestError, load_manifest_text

MANIFEST = """[package]
name = "tools"
version = "1.0.0"
[commands]
"devel nested check" = { program = "tools/main::main" }
"devel nested inspect" = { program = "tools/main::main" }
"devel releases" = {}
"devel releases inspect" = { program = "tools/main::main" }
[commands.devel]
description = "Development tasks"
help = "Choose a development workflow."
[commands.devel.review]
program = "tools/main::main"
description = "Review changes"
help = "Inspect changes before publishing."
[aliases]
dev = "devel"
rev = "devel review"
"""


@pytest.fixture
def command_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "tools"
    (root / "src").mkdir(parents=True)
    (root / "package.toml").write_text(MANIFEST)
    (root / "src" / "main.agl").write_text(
        "import std/config\n"
        '@doc("Review the selected subject.")\n'
        'program def main(@doc("Subject to inspect.") subject: text) -> unit =\n'
        "  print subject\n"
        "  print std/config::strict-json\n"
    )
    install_directory(root, home=tmp_path / "home")
    return tmp_path / "home" / ".agm" / "config.toml"


@pytest.mark.parametrize("path", ["devel", "dev", "devel nested", "dev nested"])
@pytest.mark.parametrize("help_form", ["bare", "flag", "help"])
def test_group_help_lists_descendants(command_package: Path, path: str, help_form: str) -> None:
    args = path.split()
    if help_form == "flag":
        args.append("--help")
    elif help_form == "help":
        args.insert(0, "help")
    result = CliRunner().invoke(get_command(app), args, catch_exceptions=False)
    assert result.exit_code == 0
    assert "check" in result.output
    assert "Review the selected subject." in result.output
    if "nested" not in path:
        assert "review" in result.output
        assert "releases" in result.output
        assert "Choose a development workflow." in result.output
        assert "Review changes" in result.output


@pytest.mark.parametrize("path", ["devel review", "dev review", "rev"])
def test_leaf_help_uses_manifest_help(command_package: Path, path: str) -> None:
    result = CliRunner().invoke(get_command(app), [*path.split(), "--help"])
    assert result.exit_code == 0
    assert "Inspect changes before publishing." in result.output
    assert "--subject" in result.output
    assert "Review the selected subject." in result.output
    assert "Subject to inspect." in result.output
    assert path in result.output


@pytest.mark.parametrize("section", ["rev", "dev.review", "devel.review"])
@pytest.mark.parametrize("path", ["rev", "dev review", "devel review", "exec tools/main::main"])
def test_alias_config_applies_to_every_invocation(
    command_package: Path, section: str, path: str
) -> None:
    command_package.write_text(f'[{section}]\nsubject = "configured"\nstrict-json = true\n')
    result = CliRunner().invoke(get_command(app), path.split(), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert result.output == "configured\ntrue\n"


@pytest.mark.parametrize(
    "extra",
    [
        '[aliases]\ndev = "absent"',
        "[aliases]\ndev = 1",
        '[aliases]\nhelp = "devel review"',
        '[aliases]\n"bad  path" = "devel review"',
        '[aliases]\ndevel = "devel review"',
        '[aliases]\ndev = "rev"\nrev = "dev"',
        '[commands.lonely]\nhelp = "No children"',
    ],
)
def test_invalid_aliases_and_empty_groups_are_rejected(extra: str) -> None:
    base = MANIFEST.split("[commands]")[0]
    base += '[commands]\n"devel review" = { program = "tools/main::main" }\n'
    with pytest.raises(ManifestError):
        load_manifest_text(base + extra)


def test_alias_config_conflicts_and_layer_precedence(command_package: Path) -> None:
    command_package.write_text('[rev]\nsubject = "home"\nstrict-json = true\n')
    local = Path.cwd() / ".agm"
    local.mkdir()
    (local / "config.toml").write_text('[devel.review]\nsubject = "workspace"\n')
    runner = CliRunner()
    result = runner.invoke(get_command(app), ["rev"], catch_exceptions=False)
    assert result.exit_code == 0
    assert result.output == "workspace\ntrue\n"
    (local / "config.toml").write_text(
        '[rev]\nsubject = "alias"\n[devel.review]\nsubject = "canonical"\n'
    )
    result = runner.invoke(get_command(app), ["rev"], catch_exceptions=False)
    assert result.exit_code != 0


def test_alias_metadata_survives_distribution(command_package: Path) -> None:
    from agm.packages.distribution import normalized_manifest

    manifest = load_manifest_text(MANIFEST)
    assert load_manifest_text(normalized_manifest(manifest).decode()) == manifest
    result = CliRunner().invoke(get_command(app), ["pkg", "info", "tools"])
    assert result.exit_code == 0
    assert "dev: devel" in result.output
    assert "rev: devel review" in result.output


def test_group_rejects_unknown_subcommands(command_package: Path) -> None:
    result = CliRunner().invoke(get_command(app), ["dev", "missing"])
    assert result.exit_code != 0


def test_group_help_remains_available_when_source_is_missing(command_package: Path) -> None:
    (command_package.parent / "packages" / "tools" / "1.0.0" / "src" / "main.agl").unlink()
    result = CliRunner().invoke(get_command(app), ["dev", "--help"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "nested check" in result.output
    assert "review" in result.output
    assert "Choose a development workflow." in result.output


@pytest.mark.parametrize("prefix, expected", [("--h", ["--help"]), ("--x", [])])
def test_group_option_completion(command_package: Path, prefix: str, expected: list[str]) -> None:
    from agm.completion import registered_command_param_completion

    assert [item.value for item in registered_command_param_completion(["dev"], prefix)] == expected


def test_unknown_command_help_is_rejected(command_package: Path) -> None:
    result = CliRunner().invoke(get_command(app), ["help", "absent"])
    assert result.exit_code != 0


def test_alias_can_target_an_implicit_group() -> None:
    from agm.packages.manifest import command_paths_for_program, expanded_commands

    manifest = load_manifest_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n'
        '[commands]\n"devel review" = { program = "tools/main::main" }\n'
        '[aliases]\ndev = "devel"\n'
    )
    assert expanded_commands(manifest)["dev"].program is None
    assert command_paths_for_program(manifest, "tools/main::main") == (
        ("dev", "review"),
        ("devel", "review"),
    )
