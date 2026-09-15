"""Tests for installed package command dispatch and references."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import semver
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
import agm.cli_dispatch as dispatch
from agm.cli_support.args import ExecArgs
from agm.config.context import ConfigContext
from agm.packages.activation import (
    ActivationIndex,
    ActivePackage,
    CommandRegistration,
    PackageActivationError,
    write_activation_index,
)
from agm.packages.install import install_directory
from agm.packages.layout import MODULE_TREE_DIRNAME
from agm.packages.manifest import CommandSpec, PackageManifest
from agm.packages.model import PackageInfo
from agm.packages.record import write_record
from tests._package_helpers import write_installed_package

if TYPE_CHECKING:
    from agm.agl.runtime.types import ProgramDeclInfo


def registered_help(
    path_name: str, registration: CommandRegistration, *, program: "ProgramDeclInfo | None"
) -> str:
    """Render one registered command's help from an already-discovered program."""
    from agm.cli_support.program_options import REGISTERED_RESERVED_FLAGS, program_command_for

    return dispatch.registered_command_help(
        path_name,
        registration,
        program=program,
        command=program_command_for(program, REGISTERED_RESERVED_FLAGS),
    )


def invoke(runner: CliRunner, argv: list[str], *, env: dict[str, str] | None = None) -> Result:
    return runner.invoke(
        get_command(cli.app), argv, prog_name="agm", catch_exceptions=False, env=env
    )


def test_unexpected_command_resolution_errors_are_not_treated_as_registered_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dispatch.TyperGroup,
        "resolve_command",
        lambda *_: (_ for _ in ()).throw(RuntimeError("unexpected failure")),
    )

    with pytest.raises(RuntimeError, match="unexpected failure"):
        invoke(CliRunner(), ["unknown"])


def test_registered_command_resolution_prefers_the_longest_path() -> None:
    from agm.cli_dispatch import resolve_registered_command

    commands = {
        "tools": CommandRegistration("tools", "tools/review::main"),
        "tools lint": CommandRegistration("tools", "tools/lint::main"),
    }

    resolution = resolve_registered_command(["tools", "lint", "--strict"], commands)

    assert resolution is not None
    assert resolution.registration == commands["tools lint"]
    assert resolution.trailing_args == ("--strict",)


def test_registered_command_resolution_tries_shorter_paths_after_a_longer_miss() -> None:
    from agm.cli_dispatch import resolve_registered_command

    commands = {
        "tools": CommandRegistration("tools", "tools/review::main"),
        "tools lint": CommandRegistration("tools", "tools/lint::main"),
    }

    resolution = resolve_registered_command(["tools", "--verbose"], commands)

    assert resolution is not None
    assert resolution.registration == commands["tools"]
    assert resolution.trailing_args == ("--verbose",)


def test_command_index_loader_reads_the_active_index(tmp_path: Path) -> None:
    from agm.cli_dispatch import load_command_index

    home = tmp_path / "home"
    assert load_command_index(home=home, proj_dir=None, cwd=tmp_path).commands == {}


def test_command_index_loader_uses_project_selected_package_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.packages.activation as activation

    home = tmp_path / "home"
    pinned_package = PackageInfo(
        tmp_path / "pinned",
        PackageManifest(
            "tools",
            semver.Version.parse("2.0.0"),
            commands={"new": CommandSpec("tools/new::main")},
        ),
    )
    monkeypatch.setattr(
        activation,
        "load_activation_index",
        lambda **_: ActivationIndex(
            packages={"tools": ActivePackage(semver.Version.parse("1.0.0"))},
            commands={"old": CommandRegistration("tools", "tools/old::main")},
        ),
    )
    monkeypatch.setattr(activation, "_selected_active_packages", lambda **_: (pinned_package,))

    index = dispatch.load_command_index(home=home, proj_dir=tmp_path, cwd=tmp_path)

    assert index.commands == {"new": CommandRegistration("tools", "tools/new::main")}


def test_registered_command_dispatches_trailing_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.commands.exec_program as exec_program

    context = ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)
    index = ActivationIndex(
        packages={"tools": ActivePackage(semver.Version.parse("1.0.0"))},
        commands={"tools lint": CommandRegistration("tools", "tools/lint::main")},
    )
    monkeypatch.setattr(dispatch, "current_config_context", lambda: context)
    monkeypatch.setattr(dispatch, "load_command_index", lambda **_: index)
    calls: list[tuple[str, list[str], str, str]] = []
    monkeypatch.setattr(
        exec_program,
        "run_registered",
        lambda program, argument_tokens, *, package, command_path: calls.append(
            (program, argument_tokens, package, command_path)
        ),
    )

    result = invoke(CliRunner(), ["tools", "lint", "--level", "strict"])

    assert result.exit_code == 0
    assert calls == [("tools/lint::main", ["--level", "strict"], "tools", "tools lint")]


def test_plain_registered_command_does_not_discover_during_outer_parsing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.commands.exec_program as exec_program

    context = ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)
    index = ActivationIndex(
        packages={"tools": ActivePackage(semver.Version.parse("1.0.0"))},
        commands={"tools lint": CommandRegistration("tools", "tools/lint::main")},
    )
    monkeypatch.setattr(dispatch, "current_config_context", lambda: context)
    monkeypatch.setattr(dispatch, "load_command_index", lambda **_: index)
    monkeypatch.setattr(
        exec_program,
        "registered_program_declaration",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected discovery")),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        exec_program,
        "run_registered",
        lambda _program, argument_tokens, **_kwargs: calls.append(argument_tokens),
    )

    result = invoke(CliRunner(), ["tools", "lint", "input"])

    assert result.exit_code == 0
    assert calls == [["input"]]


def test_registered_command_treats_only_standalone_dry_run_as_global(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.commands.exec_program as exec_program
    from agm.core import dry_run

    context = ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)
    index = ActivationIndex(
        commands={"tools lint": CommandRegistration("tools", "tools/lint::main")}
    )
    monkeypatch.setattr(dispatch, "current_config_context", lambda: context)
    monkeypatch.setattr(dispatch, "load_command_index", lambda **_: index)
    calls: list[tuple[list[str], bool]] = []
    monkeypatch.setattr(
        exec_program,
        "run_registered",
        lambda _program, argument_tokens, **_kwargs: calls.append(
            (argument_tokens, dry_run.enabled())
        ),
    )

    value_result = invoke(CliRunner(), ["tools", "lint", "--level=--dry-run"])
    flag_result = invoke(CliRunner(), ["tools", "lint", "--level", "strict", "--dry-run"])

    assert value_result.exit_code == 0
    assert flag_result.exit_code == 0
    assert calls == [(["--level=--dry-run"], False), (["--level", "strict"], True)]


def test_registered_command_preserves_a_host_looking_program_option_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.commands.exec_program as exec_program
    from agm.cli_support.program_discovery import discover_program_declarations_from_source

    context = ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)
    index = ActivationIndex(
        commands={"tools lint": CommandRegistration("tools", "tools/lint::main")}
    )
    monkeypatch.setattr(dispatch, "current_config_context", lambda: context)
    monkeypatch.setattr(dispatch, "load_command_index", lambda **_: index)
    (program,) = discover_program_declarations_from_source(
        "program def main(message: text) -> unit = ()"
    )
    monkeypatch.setattr(exec_program, "registered_program_declaration", lambda *_a, **_k: program)
    calls: list[list[str]] = []

    def run_registered(_program: str, argument_tokens: list[str], **_kwargs: object) -> None:
        calls.append(argument_tokens)

    monkeypatch.setattr(exec_program, "run_registered", run_registered)

    result = invoke(CliRunner(), ["tools", "lint", "--message", "--dry-run"])

    assert result.exit_code == 0
    assert calls == [["--message", "--dry-run"]]


def test_ambiguous_registered_value_reuses_static_pipeline_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agm.agl.matchcompile.stage import MatchCompiledProgram
    from agm.agl.pipeline import PipelineDriver, PreparedProgram, ProgramDiscovery

    home = tmp_path / "home"
    write_installed_package(
        home,
        "tools",
        source="program def main(message: text) -> unit = ()\n",
        commands={"tools run": "tools/main::main"},
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        dispatch,
        "load_command_index",
        lambda **_: ActivationIndex(
            commands={"tools run": CommandRegistration("tools", "tools/main::main")}
        ),
    )
    real_discover = PipelineDriver.discover_programs
    discoveries = 0

    def counting_discover(
        self: PipelineDriver,
        prepared: PreparedProgram,
        *,
        compiled: MatchCompiledProgram | None = None,
    ) -> ProgramDiscovery:
        nonlocal discoveries
        discoveries += 1
        return real_discover(self, prepared, compiled=compiled)

    monkeypatch.setattr(PipelineDriver, "discover_programs", counting_discover)

    result = invoke(CliRunner(), ["tools", "run", "--message", "--dry-run"])

    assert result.exit_code == 0
    assert discoveries == 1


def test_registered_command_help_does_not_dispatch_program(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.commands.exec_program as exec_program
    from agm.cli_support.program_discovery import discover_program_declarations_from_source

    context = ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)
    index = ActivationIndex(
        commands={
            "tools lint": CommandRegistration("tools", "tools/lint::main", "Lint package inputs")
        }
    )
    monkeypatch.setattr(dispatch, "current_config_context", lambda: context)
    monkeypatch.setattr(dispatch, "load_command_index", lambda **_: index)
    (program,) = discover_program_declarations_from_source(
        "program def main(level: text, verbose: bool, message: text) -> unit = ()"
    )
    monkeypatch.setattr(exec_program, "registered_program_declaration", lambda *_a, **_k: program)
    calls: list[object] = []
    monkeypatch.setattr(exec_program, "run_registered", lambda *args, **kwargs: calls.append(args))

    result = invoke(CliRunner(), ["tools", "lint", "--help"])
    short_result = invoke(CliRunner(), ["tools", "lint", "-h"])
    value_option_result = invoke(CliRunner(), ["tools", "lint", "--level", "strict", "-h"])
    bool_option_result = invoke(CliRunner(), ["tools", "lint", "--verbose", "-h"])
    value_result = invoke(CliRunner(), ["tools", "lint", "--message", "-h"])

    assert result.exit_code == 0
    assert short_result.exit_code == 0
    assert value_option_result.exit_code == 0
    assert bool_option_result.exit_code == 0
    assert value_result.exit_code == 0
    assert "agm tools lint" in result.output
    assert "Lint package inputs" in result.output
    assert "--dry-run" in result.output
    assert "agm tools lint" in value_option_result.output
    assert "agm tools lint" in bool_option_result.output
    assert calls == [("tools/lint::main", ["--message", "-h"])]


def test_registered_command_help_recognizes_program_value_argument_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``-h`` disambiguation and help rendering also cover a program's own
    value parameters: a bare ``-h`` is short help, but ``-h`` supplied as a
    value-taking flag's own VALUE is not.
    """
    import agm.commands.exec_program as exec_program
    from agm.cli_support.program_discovery import discover_program_declarations_from_source

    context = ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)
    index = ActivationIndex(
        commands={"tools lint": CommandRegistration("tools", "tools/lint::main")}
    )
    monkeypatch.setattr(dispatch, "current_config_context", lambda: context)
    monkeypatch.setattr(dispatch, "load_command_index", lambda **_: index)
    (program,) = discover_program_declarations_from_source(
        "program def main(tag: text) -> unit = print tag"
    )
    monkeypatch.setattr(exec_program, "registered_program_declaration", lambda *_a, **_k: program)
    calls: list[object] = []
    monkeypatch.setattr(exec_program, "run_registered", lambda *args, **kwargs: calls.append(args))

    bare_short = invoke(CliRunner(), ["tools", "lint", "-h"])
    value_short = invoke(CliRunner(), ["tools", "lint", "--tag", "-h"])

    assert bare_short.exit_code == 0
    assert "agm tools lint" in bare_short.output
    assert "--tag" in bare_short.output
    assert value_short.exit_code == 0
    assert calls == [("tools/lint::main", ["--tag", "-h"])]


def test_help_command_renders_registered_command_help(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    context = ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)
    index = ActivationIndex(
        commands={
            "tools lint": CommandRegistration("tools", "tools/lint::main", "Lint package inputs")
        }
    )
    monkeypatch.setattr(dispatch, "current_config_context", lambda: context)
    monkeypatch.setattr(dispatch, "load_command_index", lambda **_: index)

    result = invoke(CliRunner(), ["help", "tools", "lint"])

    assert result.exit_code == 0
    assert "agm tools lint" in result.output
    assert "Lint package inputs" in result.output


def test_registered_command_help_degrades_when_program_discovery_fails() -> None:

    text = registered_help(
        "tools lint", CommandRegistration("tools", "tools/lint::main"), program=None
    )

    assert "Run the registered AgL program." in text
    assert "--level" not in text

    from agm.cli_support.program_discovery import discover_program_declarations_from_source

    (program,) = discover_program_declarations_from_source(
        "program def main(level: text) -> unit = ()"
    )

    text = registered_help(
        "tools lint", CommandRegistration("tools", "tools/lint::main"), program=program
    )

    assert "--level" in text


def test_registered_command_help_omits_program_arguments_on_a_reservation_collision() -> None:
    """A value parameter colliding with a reserved flag (e.g. ``help``) renders
    no parameter entries at all: ``program_command_for`` degrades the whole
    command to ``None`` on a collision.
    """
    from agm.cli_support.program_discovery import discover_program_declarations_from_source

    (program,) = discover_program_declarations_from_source(
        "program def main(help: text) -> unit = print help"
    )

    text = registered_help(
        "tools lint", CommandRegistration("tools", "tools/lint::main"), program=program
    )

    assert "help TEXT" not in text


def test_registered_command_help_includes_manifest_description_and_program_doc() -> None:
    """A command description introduces the program's own ``@doc``."""
    from agm.cli_support.program_discovery import discover_program_declarations_from_source

    (program,) = discover_program_declarations_from_source(
        '@doc("Program prose.")\nprogram def main() -> unit = ()'
    )

    described = registered_help(
        "tools lint",
        CommandRegistration("tools", "tools/lint::main", "Manifest prose."),
        program=program,
    )
    undescribed = registered_help(
        "tools lint", CommandRegistration("tools", "tools/lint::main"), program=program
    )

    assert "Manifest prose." in described
    assert "Program prose." in described
    assert "Program prose." in undescribed


def test_registered_command_help_omits_a_hidden_parameter() -> None:
    from agm.cli_support.program_discovery import discover_program_declarations_from_source

    (program,) = discover_program_declarations_from_source(
        'program def main(level: text = "a", @opt-hidden debug-mode: bool = false) -> unit = ()'
    )

    text = registered_help(
        "tools lint", CommandRegistration("tools", "tools/lint::main"), program=program
    )

    assert "--level" in text
    assert "--debug-mode" not in text


def test_registered_command_help_usage_line_reflects_the_program_signature() -> None:
    """The rendered usage line names the invoking command and the program's
    own positional slots and options, not the raw ``program def`` declaration
    path.
    """
    from agm.cli_support.program_discovery import discover_program_declarations_from_source

    (program,) = discover_program_declarations_from_source(
        'program def main(@arg-pos name: text, @arg-std tag: text = "default") -> unit = ()'
    )

    text = registered_help(
        "tools greet",
        CommandRegistration("tools", "tools/greet::main"),
        program=program,
    )

    first_line = text.splitlines()[0]
    assert "agm tools greet" in first_line
    assert "<name>" in first_line
    assert "[OPTIONS]" in first_line
    assert "main" not in first_line


def test_registered_command_help_renders_no_contentless_sections_for_a_parameterless_program() -> (
    None
):
    """A registered command backed by a parameterless ``program def`` renders a
    plain usage line and no positional slots.
    """
    from agm.cli_support.program_discovery import discover_program_declarations_from_source

    (program,) = discover_program_declarations_from_source("program def main() -> unit = ()")

    text = registered_help(
        "tools greet",
        CommandRegistration("tools", "tools/greet::main"),
        program=program,
    )

    first_line = text.splitlines()[0]
    assert "agm tools greet" in first_line
    assert "<" not in first_line


def test_registered_command_program_option_error_renders_shared_usage_help(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A CLI parse failure against a selected program's own value parameters
    renders through the same ``registered_command_help`` rendering, including
    the program's own usage line.
    """

    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    module = package_root / "src" / "greet.agl"
    module.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n\n'
        '[commands]\n"tools greet" = { program = "tools/greet::main", '
        'description = "Greet someone" }\n',
        encoding="utf-8",
    )
    module.write_text(
        "program def main(@arg-pos name: text) -> unit = print name\n", encoding="utf-8"
    )
    write_record(package_root)
    write_activation_index(
        ActivationIndex({"tools": ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        dispatch,
        "load_command_index",
        lambda **_: ActivationIndex(
            commands={
                "tools greet": CommandRegistration("tools", "tools/greet::main", "Greet someone")
            }
        ),
    )

    result = invoke(CliRunner(), ["tools", "greet", "alice", "--nope", "x"])

    assert result.exit_code == 1
    err = result.output
    assert "--nope" in err
    assert "agm tools greet" in err
    assert "<name>" in err
    assert "Greet someone" in err


def test_registered_command_help_flag_bundled_into_a_short_group_renders_help(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A help flag only the program's own parser can see still renders its help.

    ``-vh`` is a short group whose last letter is the help flag; the dispatch
    layer cannot read it, so the request surfaces while the program's own
    command parses and renders there.
    """

    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    module = package_root / "src" / "greet.agl"
    module.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n\n'
        '[commands]\n"tools greet" = { program = "tools/greet::main", '
        'description = "Greet someone" }\n',
        encoding="utf-8",
    )
    module.write_text(
        'program def main(@opt-short("v") verbose: bool = false) -> unit = print verbose\n',
        encoding="utf-8",
    )
    write_record(package_root)
    write_activation_index(
        ActivationIndex({"tools": ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        dispatch,
        "load_command_index",
        lambda **_: ActivationIndex(
            commands={
                "tools greet": CommandRegistration("tools", "tools/greet::main", "Greet someone")
            }
        ),
    )

    result = invoke(CliRunner(), ["tools", "greet", "-vh"])

    assert result.exit_code == 0
    assert "Greet someone" in result.output
    assert "--verbose" in result.output


def test_registered_command_binds_a_negated_bool_value_argument(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A registered command's ``--no-<name>`` negation actually binds
    ``false`` to the referenced program's own bool value parameter, not
    merely offered by completion.
    """

    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    module = package_root / "src" / "run.agl"
    module.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n\n'
        '[commands]\n"tools run" = { program = "tools/run::main" }\n',
        encoding="utf-8",
    )
    module.write_text(
        "program def main(verbose: bool = true) -> unit = print verbose\n", encoding="utf-8"
    )
    write_record(package_root)
    write_activation_index(
        ActivationIndex({"tools": ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        dispatch,
        "load_command_index",
        lambda **_: ActivationIndex(
            commands={"tools run": CommandRegistration("tools", "tools/run::main")}
        ),
    )

    result = invoke(CliRunner(), ["tools", "run", "--no-verbose"])

    assert result.exit_code == 0
    assert result.stdout == "false\n"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["tools", "run", "--", "--odd"], "--odd|y\n"),
        (["tools", "run", "--", "--", "--odd"], "--|--odd\n"),
    ],
)
def test_registered_command_reaches_the_programs_end_of_options_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str], expected: str
) -> None:
    """A registered command inherits ``agm exec``'s single-``--`` rule.

    AGM's own parser keeps the marker the reader wrote, so one bare ``--``
    is the program's own end-of-options marker and a flag-shaped positional
    value reaches it behind that single marker. A second marker is then an
    ordinary positional value of its own.
    """

    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    module = package_root / "src" / "run.agl"
    module.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n\n'
        '[commands]\n"tools run" = { program = "tools/run::main" }\n',
        encoding="utf-8",
    )
    module.write_text(
        'program def main(@arg-pos who: text = "x", @arg-pos rest: text = "y") -> unit =\n'
        '  print "%{who}|%{rest}"\n',
        encoding="utf-8",
    )
    write_record(package_root)
    write_activation_index(
        ActivationIndex({"tools": ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        dispatch,
        "load_command_index",
        lambda **_: ActivationIndex(
            commands={"tools run": CommandRegistration("tools", "tools/run::main")}
        ),
    )

    result = invoke(CliRunner(), argv)

    assert result.exit_code == 0
    assert result.stdout == expected


def test_registered_command_program_may_claim_exec_only_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A registered command reserves only the flags it declares itself, not
    ``agm exec``'s: parameters spelled ``--module-path`` or ``-p`` bind, and its
    help lists them.
    """
    home = tmp_path / "home"
    write_installed_package(
        home,
        "tools",
        source=(
            'program def main(module-path: text = "a", @opt-short("p") path: text = "b")\n'
            "  -> unit =\n"
            '  print "%{module-path}|%{path}"\n'
        ),
        commands={"tools run": "tools/main::main"},
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        dispatch,
        "load_command_index",
        lambda **_: ActivationIndex(
            commands={"tools run": CommandRegistration("tools", "tools/main::main")}
        ),
    )

    result = invoke(CliRunner(), ["tools", "run", "--module-path", "lib", "-p", "src"])
    help_result = invoke(CliRunner(), ["tools", "run", "--help"])

    assert result.exit_code == 0
    assert result.stdout == "lib|src\n"
    assert help_result.exit_code == 0
    assert "--module-path" in help_result.output
    assert "-p" in help_result.output


def test_registered_help_reuses_discovery_for_a_host_shaped_program_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    write_installed_package(
        home,
        "tools",
        source="program def main(tag: text) -> unit = ()\n",
        commands={"tools run": "tools/main::main"},
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        dispatch,
        "load_command_index",
        lambda **_: ActivationIndex(
            commands={"tools run": CommandRegistration("tools", "tools/main::main")}
        ),
    )

    result = invoke(CliRunner(), ["tools", "run", "--tag", "--dry-run", "--help"])

    assert result.exit_code == 0
    assert "--tag" in result.output


def test_registered_command_help_returns_false_when_index_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        dispatch,
        "current_config_context",
        lambda: ConfigContext(tmp_path / "home", None, tmp_path),
    )
    monkeypatch.setattr(
        dispatch,
        "load_command_index",
        lambda **_: (_ for _ in ()).throw(ValueError("bad index")),
    )

    assert not dispatch.print_registered_command_help(["tools", "lint"])


def test_unknown_command_without_registered_entry_keeps_click_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    monkeypatch.setattr(
        dispatch,
        "current_config_context",
        lambda: ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path),
    )
    monkeypatch.setattr(dispatch, "load_command_index", lambda **_: ActivationIndex())

    result = invoke(CliRunner(), ["not-a-command"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_builtin_commands_do_not_load_the_package_index(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setattr(
        dispatch,
        "load_command_index",
        lambda **_: (_ for _ in ()).throw(AssertionError("builtins must not read the index")),
    )

    result = invoke(CliRunner(), ["pkg"])

    assert result.exit_code == 0


def test_help_overview_appends_registered_commands(tmp_path: Path) -> None:
    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    package_root.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        """[package]
name = "tools"
version = "1.0.0"

[commands]
"tools lint" = { program = "tools/lint::main", description = "Lint package inputs" }
""",
        encoding="utf-8",
    )
    index = ActivationIndex(
        packages={"tools": ActivePackage(semver.Version.parse("1.0.0"))},
        commands={
            "tools lint": CommandRegistration("tools", "tools/lint::main", "Lint package inputs")
        },
    )
    write_record(package_root)
    write_activation_index(index, home=home)

    result = invoke(CliRunner(), ["help"], env={"HOME": str(home)})

    assert result.exit_code == 0
    assert "Registered commands:" in result.stdout
    assert "tools lint" in result.stdout
    assert "Lint package inputs" in result.stdout


def test_help_overview_degrades_when_the_command_index_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(
        dispatch,
        "load_command_index",
        lambda **_: (_ for _ in ()).throw(ValueError("bad index")),
    )

    result = invoke(CliRunner(), ["help"])

    assert result.exit_code == 0
    assert "Registered commands:" not in result.stdout


def test_exec_installed_reference_preserves_all_file_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_support.exec_target as exec_target
    import agm.commands.exec as exec_command
    import agm.commands.exec_program as exec_program

    root = tmp_path / "tools"
    module = root / "src" / "review.agl"
    module.parent.mkdir(parents=True)
    module.write_text("program def main() -> unit = ()\n", encoding="utf-8")
    package = PackageInfo(root, PackageManifest("tools", semver.Version.parse("1.0.0")))
    context = ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)
    monkeypatch.setattr(exec_program, "current_config_context", lambda: context)
    monkeypatch.setattr(exec_target, "select_active_packages", lambda **_: (package,))
    calls: list[ExecArgs] = []

    def fake_run(args: ExecArgs, **_: object) -> None:
        calls.append(args)

    monkeypatch.setattr(exec_program, "run", fake_run)

    args = ExecArgs(
        file="tools/review::main",
        strict_json=True,
        no_log=True,
        log_file="trace.jsonl",
        argument_tokens=["--subject", "changes"],
        log=True,
        module_paths=["modules"],
        no_stdlib=True,
        max_iters=3,
        max_call_depth=4,
        timeout="5s",
        no_timeout=True,
        no_log_file=True,
        default_agent='AgentCommand("fake")',
    )

    exec_command.run(args)

    assert calls == [replace(args, file=str(module), program="main")]


def test_exec_prefers_an_existing_file_path_containing_a_reference_separator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.commands.exec as exec_command
    import agm.commands.exec_program as exec_program

    agl_file = tmp_path / "tools::review.agl"
    agl_file.write_text("program def main() -> unit = ()\n", encoding="utf-8")
    calls: list[ExecArgs] = []
    monkeypatch.setattr(exec_program, "run", lambda args, **_: calls.append(args))

    args = ExecArgs(file=str(agl_file), strict_json=None, no_log=False, log_file=None)
    exec_command.run(args)

    assert calls == [args]


@pytest.mark.parametrize(
    "argv",
    (["exec", "tools/main::main"], ["tools", "run"]),
    ids=("installed-reference", "registered-command"),
)
def test_immutable_execution_uses_the_pinned_dependency_not_a_vendored_path_source(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    config = project / ".agm"
    config.mkdir(parents=True)
    (config / "config.toml").write_text('[packages]\nshared = "1.0.0"\n', encoding="utf-8")

    relocated_store = tmp_path / "relocated-packages"
    relocated_store.mkdir()
    logical_store = home / ".agm" / "packages"
    logical_store.parent.mkdir(parents=True)
    logical_store.symlink_to(relocated_store, target_is_directory=True)

    def write_shared(root: Path, version: str, label: str) -> None:
        module = root / "src" / "value.agl"
        module.parent.mkdir(parents=True)
        (root / "package.toml").write_text(
            f'[package]\nname = "shared"\nversion = "{version}"\n', encoding="utf-8"
        )
        module.write_text(f'def label() -> text = "{label}"\n', encoding="utf-8")

    pinned_shared = relocated_store / "shared" / "1.0.0"
    active_shared = relocated_store / "shared" / "2.0.0"
    write_shared(pinned_shared, "1.0.0", "pinned")
    write_shared(active_shared, "2.0.0", "active")
    write_record(pinned_shared)
    write_record(active_shared)

    tools = relocated_store / "tools" / "1.0.0"
    module = tools / "src" / "main.agl"
    module.parent.mkdir(parents=True)
    (tools / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n\n'
        '[dependencies]\nshared = { version = "1", path = "vendor/shared" }\n\n'
        '[commands."tools run"]\nprogram = "tools/main::main"\n',
        encoding="utf-8",
    )
    module.write_text(
        "import shared/value\nprogram def main() -> unit = print shared/value::label()\n",
        encoding="utf-8",
    )
    vendored_shared = tools / "vendor" / "shared"
    write_shared(vendored_shared, "3.0.0", "vendored")
    write_record(tools)

    write_activation_index(
        ActivationIndex(
            packages={
                "tools": ActivePackage(semver.Version.parse("1.0.0")),
                "shared": ActivePackage(semver.Version.parse("2.0.0")),
            },
            commands={"tools run": CommandRegistration("tools", "tools/main::main")},
        ),
        home=home,
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    result = invoke(CliRunner(), argv)

    assert result.exit_code == 0, result.output
    assert result.stdout == "pinned\n"


def test_exec_runs_an_installed_reference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    module = package_root / "src" / "review.agl"
    module.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    module.write_text(
        "program def main(level: text) -> unit = print level\n",
        encoding="utf-8",
    )
    write_record(package_root)
    write_activation_index(
        ActivationIndex({"tools": ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    monkeypatch.setenv("HOME", str(home))

    result = invoke(CliRunner(), ["exec", "tools/review::main", "--level", "set"])

    assert result.exit_code == 0
    assert result.stdout == "set\n"


def test_installed_package_dispatches_its_own_source_declared_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-editable install bakes ``@command`` registrations into the store manifest."""
    source = tmp_path / "source"
    (source / MODULE_TREE_DIRNAME).mkdir(parents=True)
    (source / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    (source / MODULE_TREE_DIRNAME / "review.agl").write_text(
        '@command("tools review")\n'
        '@description("Review changes")\n'
        "program def main(level: text) -> unit = print level\n",
        encoding="utf-8",
    )
    home = tmp_path / "home"

    install_directory(source, home=home, env={})
    monkeypatch.setenv("HOME", str(home))

    result = invoke(CliRunner(), ["tools", "review", "--level", "set"])

    assert result.exit_code == 0
    assert result.stdout == "set\n"


def test_editable_package_dispatches_its_own_source_declared_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An editable install rescans the live source, so its own ``@command`` dispatches."""
    source = tmp_path / "source"
    (source / MODULE_TREE_DIRNAME).mkdir(parents=True)
    (source / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    (source / MODULE_TREE_DIRNAME / "review.agl").write_text(
        '@command("tools review")\n'
        '@description("Review changes")\n'
        "program def main(level: text) -> unit = print level\n",
        encoding="utf-8",
    )
    home = tmp_path / "home"

    install_directory(source, home=home, env={}, editable=True)
    monkeypatch.setenv("HOME", str(home))

    result = invoke(CliRunner(), ["tools", "review", "--level", "set"])

    assert result.exit_code == 0
    assert result.stdout == "set\n"


def test_editing_an_editable_packages_source_command_path_takes_effect_without_reinstalling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Editable dispatch re-derives from the live source, so an edited ``@command`` path
    changes the dispatchable command set on the next invocation without reinstalling."""
    source = tmp_path / "source"
    (source / MODULE_TREE_DIRNAME).mkdir(parents=True)
    (source / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    module = source / MODULE_TREE_DIRNAME / "review.agl"
    module.write_text(
        '@command("tools review")\nprogram def main(level: text) -> unit = print level\n',
        encoding="utf-8",
    )
    home = tmp_path / "home"

    install_directory(source, home=home, env={}, editable=True)
    monkeypatch.setenv("HOME", str(home))

    before = invoke(CliRunner(), ["tools", "review", "--level", "set"])
    assert before.exit_code == 0

    module.write_text(
        '@command("tools inspect")\nprogram def main(level: text) -> unit = print level\n',
        encoding="utf-8",
    )

    after_old_path = invoke(CliRunner(), ["tools", "review", "--level", "set"])
    after_new_path = invoke(CliRunner(), ["tools", "inspect", "--level", "set"])

    assert after_old_path.exit_code != 0
    assert after_new_path.exit_code == 0
    assert after_new_path.stdout == "set\n"


def test_a_non_editable_install_does_not_pick_up_a_source_edit_made_after_installation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-editable install bakes the manifest at install time, so it never rescans the
    original source directory again."""
    source = tmp_path / "source"
    (source / MODULE_TREE_DIRNAME).mkdir(parents=True)
    (source / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    (source / MODULE_TREE_DIRNAME / "review.agl").write_text(
        "program def main(level: text) -> unit = print level\n", encoding="utf-8"
    )
    home = tmp_path / "home"

    install_directory(source, home=home, env={})
    monkeypatch.setenv("HOME", str(home))

    (source / MODULE_TREE_DIRNAME / "review.agl").write_text(
        '@command("tools review")\nprogram def main(level: text) -> unit = print level\n',
        encoding="utf-8",
    )

    result = invoke(CliRunner(), ["tools", "review", "--level", "set"])

    assert result.exit_code != 0


def test_conflicting_install_is_rejected_while_an_editable_packages_unrelated_module_is_broken(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Install-time conflict gating must see the true command set: a transient source
    discovery failure in an unrelated module of an editable package must not let a
    colliding install through and wedge the package CLI."""
    pa_source = tmp_path / "pa"
    (pa_source / MODULE_TREE_DIRNAME).mkdir(parents=True)
    (pa_source / "package.toml").write_text(
        '[package]\nname = "pa"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    (pa_source / MODULE_TREE_DIRNAME / "zap.agl").write_text(
        '@command("zap")\nprogram def main() -> unit = print "pa-zap"\n', encoding="utf-8"
    )
    home = tmp_path / "home"

    install_directory(pa_source, home=home, env={}, editable=True)
    monkeypatch.setenv("HOME", str(home))

    (pa_source / MODULE_TREE_DIRNAME / "broken.agl").write_text(
        "program def main( -> unit = ()\n", encoding="utf-8"
    )

    pb_source = tmp_path / "pb"
    (pb_source / MODULE_TREE_DIRNAME).mkdir(parents=True)
    (pb_source / "package.toml").write_text(
        '[package]\nname = "pb"\nversion = "1.0.0"\n\n'
        '[commands]\nzap = { program = "pb/main::main" }\n',
        encoding="utf-8",
    )
    (pb_source / MODULE_TREE_DIRNAME / "main.agl").write_text(
        'program def main() -> unit = print "pb-zap"\n', encoding="utf-8"
    )

    result = invoke(CliRunner(), ["pkg", "install", str(pb_source)])

    assert result.exit_code != 0

    listing = invoke(CliRunner(), ["pkg", "list"])
    assert listing.exit_code == 0
    assert "pb" not in listing.output

    (pa_source / MODULE_TREE_DIRNAME / "broken.agl").unlink()

    dispatched = invoke(CliRunner(), ["zap"])
    assert dispatched.exit_code == 0
    assert dispatched.stdout == "pa-zap\n"


def test_pkg_list_survives_an_unparsable_module_in_an_editable_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A syntax error in one module is an editable package's normal, transient state while
    it is being edited; that must never make an unrelated built-in command fail."""
    source = tmp_path / "source"
    (source / MODULE_TREE_DIRNAME).mkdir(parents=True)
    (source / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    (source / MODULE_TREE_DIRNAME / "review.agl").write_text(
        '@command("tools review")\nprogram def main() -> unit = ()\n', encoding="utf-8"
    )
    home = tmp_path / "home"

    install_directory(source, home=home, env={}, editable=True)
    monkeypatch.setenv("HOME", str(home))

    (source / MODULE_TREE_DIRNAME / "broken.agl").write_text(
        "program def main( -> unit = ()\n", encoding="utf-8"
    )

    result = invoke(CliRunner(), ["pkg", "list"])

    assert result.exit_code == 0
    assert "tools" in result.output


def test_dispatch_and_unrelated_commands_survive_a_broken_module_in_an_editable_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The tolerance is for reads only, but it must actually cover every read: a broken,
    unrelated module in one editable package must not stop another package's registered
    command from dispatching, nor a built-in command that never touches that package."""
    tools_source = tmp_path / "tools"
    (tools_source / MODULE_TREE_DIRNAME).mkdir(parents=True)
    (tools_source / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    (tools_source / MODULE_TREE_DIRNAME / "review.agl").write_text(
        '@command("tools review")\nprogram def main() -> unit = ()\n', encoding="utf-8"
    )
    helper_source = tmp_path / "helper"
    (helper_source / MODULE_TREE_DIRNAME).mkdir(parents=True)
    (helper_source / "package.toml").write_text(
        '[package]\nname = "helper"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    (helper_source / MODULE_TREE_DIRNAME / "run.agl").write_text(
        '@command("helper run")\nprogram def main() -> unit = print "ran"\n', encoding="utf-8"
    )
    home = tmp_path / "home"

    install_directory(tools_source, home=home, env={}, editable=True)
    install_directory(helper_source, home=home, env={}, editable=False)
    monkeypatch.setenv("HOME", str(home))

    (tools_source / MODULE_TREE_DIRNAME / "broken.agl").write_text(
        "program def main( -> unit = ()\n", encoding="utf-8"
    )

    dispatched = invoke(CliRunner(), ["helper", "run"])
    assert dispatched.exit_code == 0
    assert dispatched.stdout == "ran\n"

    # A built-in command that resolves only a single, non-editable package
    # never reaches source discovery, so it is unaffected either way.
    info = invoke(CliRunner(), ["pkg", "info", "helper"])
    assert info.exit_code == 0
    assert "helper" in info.output


def test_pkg_check_still_reports_an_unparsable_module_in_an_editable_package(
    tmp_path: Path,
) -> None:
    """Validation stays exactly as strict as it was: ``pkg check`` is where source discipline
    is enforced, so it must still fail loudly on the module activation now tolerates."""
    source = tmp_path / "source"
    (source / MODULE_TREE_DIRNAME).mkdir(parents=True)
    (source / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    (source / MODULE_TREE_DIRNAME / "review.agl").write_text(
        '@command("tools review")\nprogram def main() -> unit = ()\n', encoding="utf-8"
    )
    (source / MODULE_TREE_DIRNAME / "broken.agl").write_text(
        "program def main( -> unit = ()\n", encoding="utf-8"
    )

    result = invoke(CliRunner(), ["pkg", "check", str(source)])

    assert result.exit_code != 0
    assert "broken.agl" in result.output


def test_exec_help_for_an_installed_reference_includes_program_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    module = package_root / "src" / "review.agl"
    module.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    module.write_text(
        "program def main(level: text) -> unit = ()\n",
        encoding="utf-8",
    )
    write_record(package_root)
    write_activation_index(
        ActivationIndex({"tools": ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    monkeypatch.setenv("HOME", str(home))

    result = invoke(CliRunner(), ["exec", "tools/review::main", "--help"])

    assert result.exit_code == 0
    assert "--level" in result.output


def test_exec_program_option_overrides_an_installed_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    module = package_root / "src" / "review.agl"
    module.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    module.write_text(
        'program def main() -> unit = print "main"\n'
        'program def alternate() -> unit = print "alternate"\n',
        encoding="utf-8",
    )
    write_record(package_root)
    write_activation_index(
        ActivationIndex({"tools": ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    monkeypatch.setenv("HOME", str(home))

    result = invoke(CliRunner(), ["exec", "-p", "alternate", "tools/review::main"])

    assert result.exit_code == 0
    assert result.stdout == "alternate\n"


def test_program_argument_parse_failure_raises_a_typed_usage_error(tmp_path: Path) -> None:
    """``run`` no longer renders a usage message itself: it raises a typed error
    carrying the parse-failure message and the selected program's own
    declaration, so the CALLER (a plain ``agm exec`` invocation, or
    registered-command dispatch) can render it appropriately. See
    ``agm.commands.exec`` and ``agm.cli_dispatch``."""
    import agm.commands.exec_program as exec_program
    from agm.commands.exec_program import RegisteredProgramUsageError

    source = tmp_path / "main.agl"
    source.write_text("program def main(level: text) -> unit = ()\n", encoding="utf-8")

    with pytest.raises(RegisteredProgramUsageError) as exc_info:
        exec_program.run(
            ExecArgs(
                file=str(source),
                argument_tokens=["--unknown"],
                strict_json=None,
                no_log=False,
                log_file=None,
                no_stdlib=True,
            ),
        )

    assert "--unknown" in exc_info.value.message
    assert exc_info.value.program is not None
    assert [p.name for p in exc_info.value.program.parameters] == ["level"]


def test_registered_command_argument_error_renders_shared_usage_help(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A CLI argument-parse failure on a dispatched registered command renders
    through the same ``registered_command_help`` the ``--help``/``-h`` paths
    use, instead of a bespoke duplicate rendering."""

    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    module = package_root / "src" / "lint.agl"
    module.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n\n'
        '[commands]\n"tools lint" = { program = "tools/lint::main", '
        'description = "Lint package inputs" }\n',
        encoding="utf-8",
    )
    module.write_text(
        "@param let verbose: bool = false\nprogram def main(level: text) -> unit = ()\n",
        encoding="utf-8",
    )
    write_record(package_root)
    write_activation_index(
        ActivationIndex({"tools": ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        dispatch,
        "load_command_index",
        lambda **_: ActivationIndex(
            commands={
                "tools lint": CommandRegistration(
                    "tools", "tools/lint::main", "Lint package inputs"
                )
            }
        ),
    )

    result = invoke(CliRunner(), ["tools", "lint", "--unknown"])

    assert result.exit_code == 1
    err = result.output
    assert "--unknown" in err
    assert "agm tools lint" in err
    assert "Lint package inputs" in err
    assert "Options:" in err
    assert "--level" in err
    assert "Parameters of tools/lint" in err
    assert "--verbose" in err


def test_registered_command_argument_error_handles_no_description_or_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    module = package_root / "src" / "lint.agl"
    module.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n\n'
        '[commands]\n"tools lint" = { program = "tools/lint::main" }\n',
        encoding="utf-8",
    )
    module.write_text("program def main() -> unit = ()\n", encoding="utf-8")
    write_record(package_root)
    write_activation_index(
        ActivationIndex({"tools": ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        dispatch,
        "load_command_index",
        lambda **_: ActivationIndex(
            commands={"tools lint": CommandRegistration("tools", "tools/lint::main")}
        ),
    )

    result = invoke(CliRunner(), ["tools", "lint", "--unknown"])

    assert result.exit_code == 1
    err = result.output
    assert "agm tools lint" in err
    assert "Run the registered AgL program." in err


def test_plain_exec_argument_error_still_renders_the_base_exec_usage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A direct ``agm exec FILE`` argument-parse failure still renders the
    ordinary ``agm exec`` usage error, unaffected by the registered-command
    usage-error rendering."""
    import agm.commands.exec as exec_command

    source = tmp_path / "main.agl"
    source.write_text("program def main() -> unit = ()\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        exec_command.run(
            ExecArgs(
                file=str(source),
                argument_tokens=["--unknown"],
                strict_json=None,
                no_log=False,
                log_file=None,
                no_stdlib=True,
            )
        )

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "--unknown" in error
    assert "usage: agm exec" in error


@pytest.mark.parametrize("reference", ["not-a-reference", "1bad/main::main"])
def test_exec_rejects_malformed_installed_references(reference: str) -> None:
    import agm.commands.exec_program as exec_program

    with pytest.raises(SystemExit) as exc_info:
        exec_program.run_registered(reference, [])

    assert exc_info.value.code == 1


def test_exec_rejects_incomplete_registered_command_metadata() -> None:
    import agm.commands.exec_program as exec_program

    with pytest.raises(SystemExit) as exc_info:
        exec_program.run_registered("tools/review::main", [], package="tools")

    assert exc_info.value.code == 1


def test_exec_rejects_invalid_active_package_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_support.exec_target as exec_target
    import agm.commands.exec_program as exec_program

    monkeypatch.setattr(
        exec_program,
        "current_config_context",
        lambda: ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path),
    )
    monkeypatch.setattr(
        exec_target,
        "select_active_packages",
        lambda **_: (_ for _ in ()).throw(ValueError("broken selection")),
    )

    with pytest.raises(SystemExit) as exc_info:
        exec_program.run_registered("tools/review::main", [])

    assert exc_info.value.code == 1


def test_registered_dispatch_rejects_a_stale_cached_program(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_support.exec_target as exec_target
    import agm.commands.exec_program as exec_program

    root = tmp_path / "tools"
    module = root / "src" / "review.agl"
    module.parent.mkdir(parents=True)
    module.write_text("program def main() -> unit = ()\n", encoding="utf-8")
    package = PackageInfo(
        root,
        PackageManifest(
            "tools",
            semver.Version.parse("1.0.0"),
            commands={"review": CommandSpec("tools/review::main")},
        ),
    )
    monkeypatch.setattr(
        exec_program,
        "current_config_context",
        lambda: ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path),
    )
    monkeypatch.setattr(exec_target, "select_active_packages", lambda **_: (package,))

    with pytest.raises(SystemExit) as exc_info:
        exec_program.run_registered(
            "tools/review::other", [], package="tools", command_path="review"
        )

    assert exc_info.value.code == 1


def test_editable_registered_dispatch_uses_the_live_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_support.exec_target as exec_target
    import agm.commands.exec_program as exec_program

    root = tmp_path / "tools"
    updated = root / "src" / "updated.agl"
    updated.parent.mkdir(parents=True)
    updated.write_text("program def main() -> unit = ()\n", encoding="utf-8")
    package = PackageInfo(
        root,
        PackageManifest(
            "tools",
            semver.Version.parse("1.0.0"),
            commands={"live": CommandSpec("tools/updated::main")},
        ),
    )
    context = ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)
    index = ActivationIndex(
        packages={"tools": ActivePackage(semver.Version.parse("1.0.0"), editable=root)}
    )
    monkeypatch.setattr(exec_program, "current_config_context", lambda: context)
    monkeypatch.setattr(exec_target, "select_active_packages", lambda **_: (package,))
    monkeypatch.setattr(exec_program, "load_activation_index", lambda **_: index)
    calls: list[tuple[ExecArgs, tuple[str, ...] | None]] = []
    monkeypatch.setattr(
        exec_program,
        "run",
        lambda args, *, entry_module_segments=None, **_: calls.append(
            (args, entry_module_segments)
        ),
    )

    exec_program.run_registered("tools/main::main", [], package="tools", command_path="live")

    assert calls == [
        (
            ExecArgs(
                file=str(updated.resolve()),
                program="main",
                argument_tokens=[],
                strict_json=None,
                no_log=False,
                log_file=None,
            ),
            ("tools", "updated"),
        )
    ]


@pytest.mark.parametrize(
    "commands",
    (
        {},
        {"live": CommandSpec("not-valid")},
        {"live": CommandSpec("not-valid::main")},
        {"live": CommandSpec("other/main::main")},
    ),
)
def test_editable_registered_dispatch_rejects_invalid_live_manifest_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    commands: dict[str, CommandSpec],
) -> None:
    import agm.cli_support.exec_target as exec_target
    import agm.commands.exec_program as exec_program

    root = tmp_path / "tools"
    package = PackageInfo(
        root, PackageManifest("tools", semver.Version.parse("1.0.0"), commands=commands)
    )
    context = ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)
    index = ActivationIndex(
        packages={"tools": ActivePackage(semver.Version.parse("1.0.0"), editable=root)}
    )
    monkeypatch.setattr(exec_program, "current_config_context", lambda: context)
    monkeypatch.setattr(exec_target, "select_active_packages", lambda **_: (package,))
    monkeypatch.setattr(exec_program, "load_activation_index", lambda **_: index)

    with pytest.raises(SystemExit) as exc_info:
        exec_program.run_registered("tools/main::main", [], package="tools", command_path="live")

    assert exc_info.value.code == 1


def test_registered_dispatch_rejects_a_program_owned_by_another_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_support.exec_target as exec_target
    import agm.commands.exec_program as exec_program

    tools_root = tmp_path / "tools"
    bravo_root = tmp_path / "bravo"
    tools_module = tools_root / "src" / "review.agl"
    bravo_module = bravo_root / "src" / "review.agl"
    tools_module.parent.mkdir(parents=True)
    bravo_module.parent.mkdir(parents=True)
    bravo_module.write_text("program def main() -> unit = ()\n", encoding="utf-8")
    tools_module.symlink_to(bravo_module)
    tools = PackageInfo(
        tools_root,
        PackageManifest(
            "tools",
            semver.Version.parse("1.0.0"),
            commands={"review": CommandSpec("tools/review::main")},
        ),
    )
    bravo = PackageInfo(bravo_root, PackageManifest("bravo", semver.Version.parse("1.0.0")))
    monkeypatch.setattr(
        exec_program,
        "current_config_context",
        lambda: ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path),
    )
    monkeypatch.setattr(exec_target, "select_active_packages", lambda **_: (tools, bravo))

    with pytest.raises(SystemExit) as exc_info:
        exec_program.run_registered(
            "tools/review::main", [], package="tools", command_path="review"
        )

    assert exc_info.value.code == 1


def test_malformed_activation_index_fallback_is_a_clean_cli_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    monkeypatch.setattr(
        dispatch,
        "current_config_context",
        lambda: ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path),
    )
    monkeypatch.setattr(
        dispatch,
        "load_command_index",
        lambda **_: (_ for _ in ()).throw(PackageActivationError("malformed index")),
    )

    result = invoke(CliRunner(), ["unknown"])

    assert result.exit_code != 0
    assert "malformed index" in result.output
    assert "Traceback" not in result.output


def test_exec_rejects_missing_module_in_an_active_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_support.exec_target as exec_target
    import agm.commands.exec_program as exec_program

    package = PackageInfo(
        tmp_path / "tools", PackageManifest("tools", semver.Version.parse("1.0.0"))
    )
    monkeypatch.setattr(
        exec_program,
        "current_config_context",
        lambda: ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path),
    )
    monkeypatch.setattr(exec_target, "select_active_packages", lambda **_: (package,))

    with pytest.raises(SystemExit) as exc_info:
        exec_program.run_registered("tools/review::main", [])

    assert exc_info.value.code == 1


def test_exec_unknown_installed_reference_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_support.exec_target as exec_target
    import agm.commands.exec as exec_command
    import agm.commands.exec_program as exec_program

    monkeypatch.setattr(
        exec_program,
        "current_config_context",
        lambda: ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path),
    )
    monkeypatch.setattr(exec_target, "select_active_packages", lambda **_: ())

    with pytest.raises(SystemExit) as exc_info:
        exec_command.run(
            ExecArgs(file="missing/main::main", strict_json=None, no_log=False, log_file=None)
        )

    assert exc_info.value.code == 1


def test_registered_declaration_ignores_a_program_owned_by_another_package(tmp_path: Path) -> None:
    """A registration whose program reference names a different package than
    the registration itself is not trusted to describe that command's arguments."""
    from agm.commands.exec_program import registered_program_declaration

    write_installed_package(
        tmp_path, "tools", source="program def main(level: text) -> unit = ()\n"
    )
    context = ConfigContext(home=tmp_path, proj_dir=None, cwd=tmp_path)

    assert registered_program_declaration("tools/main::main", "tools", context=context) is not None
    assert registered_program_declaration("tools/main::main", "other", context=context) is None


def test_registered_program_declaration_finds_the_referenced_program(tmp_path: Path) -> None:
    from agm.commands.exec_program import registered_program_declaration

    context = ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)

    assert registered_program_declaration("not-a-reference", "tools", context=context) is None
    assert registered_program_declaration("tools/bad-name::main", "tools", context=context) is None
    assert registered_program_declaration("other/lint::main", "tools", context=context) is None


def test_registered_program_declaration_degrades_when_artifact_discovery_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.commands.exec_program as exec_program

    write_installed_package(tmp_path, "tools")
    context = ConfigContext(home=tmp_path, proj_dir=None, cwd=tmp_path)
    monkeypatch.setattr(exec_program, "discover_program_artifacts_for_target", lambda **_: None)

    assert (
        exec_program.registered_program_declaration("tools/main::main", "tools", context=context)
        is None
    )


def test_registered_program_declaration_finds_its_own_value_parameters(tmp_path: Path) -> None:
    """A registration's program reference resolves to its own ``program def``

    declaration, carrying its value-parameter signature — not a different
    program declared in the same module.
    """
    from agm.commands.exec_program import registered_program_declaration

    write_installed_package(
        tmp_path,
        "tools",
        source=("program def other() -> unit = ()\nprogram def main(level: text) -> unit = ()\n"),
    )
    context = ConfigContext(home=tmp_path, proj_dir=None, cwd=tmp_path)

    declaration = registered_program_declaration("tools/main::main", "tools", context=context)

    assert declaration is not None
    assert declaration.declaration_path == "main"
    assert [param.name for param in declaration.parameters] == ["level"]
    assert registered_program_declaration("tools/main::main", "other", context=context) is None


def test_registered_program_declaration_prefers_the_entry_module_over_an_import(
    tmp_path: Path,
) -> None:
    """An imported module's same-named ``program def`` never shadows the entry
    module's own declaration sharing that declaration path.
    """
    from agm.commands.exec_program import registered_program_declaration

    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    entry = package_root / "src" / "main.agl"
    helper = package_root / "src" / "helper.agl"
    entry.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    helper.write_text("program def main(other: int) -> unit = ()\n", encoding="utf-8")
    entry.write_text(
        "import tools/helper\nprogram def main(level: text) -> unit = ()\n", encoding="utf-8"
    )
    write_record(package_root)
    write_activation_index(
        ActivationIndex({"tools": ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    context = ConfigContext(home=home, proj_dir=None, cwd=tmp_path)

    declaration = registered_program_declaration("tools/main::main", "tools", context=context)

    assert declaration is not None
    assert [param.name for param in declaration.parameters] == ["level"]


def test_registered_program_declaration_degrades_when_selection_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_support.exec_target as exec_target
    from agm.commands.exec_program import registered_program_declaration

    context = ConfigContext(tmp_path / "home", None, tmp_path)
    monkeypatch.setattr(
        exec_target,
        "select_active_packages",
        lambda **_: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    assert registered_program_declaration("tools/lint::main", "tools", context=context) is None


def test_registered_command_binds_module_parameters_from_its_command_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Registered execution shares module parameter precedence with ``agm exec``."""
    import agm.commands.exec_program as exec_program

    home = tmp_path / "home"
    source = "import tools/logging\nprogram def main() -> unit = print tools/logging::verbose\n"
    module = write_installed_package(
        home,
        "tools",
        source=source,
        commands={"dev review": "tools/main::main"},
    )
    (module.parent / "logging.agl").write_text(
        '@param @opt-env("AGM_REGISTERED_VERBOSE") let verbose: bool = false\n'
        "@param let retries: int = 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        exec_program,
        "current_config_context",
        lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
    )
    monkeypatch.setenv("HOME", str(home))

    default = invoke(CliRunner(), ["dev", "review"])
    assert default.exit_code == 0
    assert default.output == "false\n"

    config = home / ".agm" / "config.toml"
    config.write_text("[tools.logging]\nverbose = true\n", encoding="utf-8")
    module_value = invoke(CliRunner(), ["dev", "review"])
    assert module_value.exit_code == 0
    assert module_value.output == "true\n"

    config.write_text(
        "[tools.logging]\nverbose = true\n\n[dev.review]\nverbose = false\n",
        encoding="utf-8",
    )

    exec_program.run_registered("tools/main::main", [], package="tools", command_path="dev review")
    assert capsys.readouterr().out == "false\n"

    exec_program.run_registered(
        "tools/main::main", ["--verbose"], package="tools", command_path="dev review"
    )
    assert capsys.readouterr().out == "true\n"

    program_value = invoke(CliRunner(), ["dev", "review"])
    assert program_value.exit_code == 0
    assert program_value.output == "false\n"

    monkeypatch.setenv("AGM_REGISTERED_VERBOSE", "true")
    environment_value = invoke(CliRunner(), ["dev", "review"])
    assert environment_value.exit_code == 0
    assert environment_value.output == "true\n"

    cli_value = invoke(CliRunner(), ["dev", "review", "--no-verbose"])
    assert cli_value.exit_code == 0
    assert cli_value.output == "false\n"

    undecodable = invoke(CliRunner(), ["dev", "review", "--tools.logging.retries", "bad"])
    assert undecodable.exit_code == 1
    assert "retries" in undecodable.output
    assert "false\n" not in undecodable.output

    help_result = invoke(CliRunner(), ["dev", "review", "--help"])
    assert help_result.exit_code == 0
    assert "Parameters of tools/logging" in help_result.output


def test_registered_command_reports_an_ambiguous_module_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    source = (
        "import tools/logging\n"
        "import tools/other\n"
        "program def main() -> unit = print tools/logging::verbose\n"
    )
    module = write_installed_package(
        home,
        "tools",
        source=source,
        commands={"dev review": "tools/main::main"},
    )
    (module.parent / "logging.agl").write_text(
        "@param let verbose: bool = false\n", encoding="utf-8"
    )
    (module.parent / "other.agl").write_text("@param let verbose: bool = false\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    result = invoke(CliRunner(), ["dev", "review", "--verbose"])

    assert result.exit_code == 1
    assert "ambiguous" in result.output.lower()
