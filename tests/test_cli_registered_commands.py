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
    from agm.cli_support.program_options import program_command_for

    return dispatch.registered_command_help(
        path_name, registration, program=program, command=program_command_for(program)
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


def test_registered_command_help_prefers_the_manifest_description_over_the_program_doc() -> None:
    """A package author's command description outranks the program's own ``@doc``."""
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
    assert "Program prose." not in described
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
    module = package_root / "tools" / "greet.agl"
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
    module = package_root / "tools" / "greet.agl"
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
    module = package_root / "tools" / "run.agl"
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


def test_registered_command_reaches_the_programs_end_of_options_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A registered command inherits ``agm exec``'s doubled-``--`` rule.

    AGM's own parser consumes one bare ``--``, so a flag-shaped positional
    value reaches the program only behind a second marker.
    """

    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    module = package_root / "tools" / "run.agl"
    module.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n\n'
        '[commands]\n"tools run" = { program = "tools/run::main" }\n',
        encoding="utf-8",
    )
    module.write_text(
        'program def main(@arg-pos who: text = "x") -> unit = print who\n', encoding="utf-8"
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

    result = invoke(CliRunner(), ["tools", "run", "--", "--", "--odd"])

    assert result.exit_code == 0
    assert result.stdout == "--odd\n"


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
    module = root / "tools" / "review.agl"
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
        agent='AgentCommand("fake")',
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
        module = root / "shared" / "value.agl"
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
    module = tools / "tools" / "main.agl"
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
    module = package_root / "tools" / "review.agl"
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


def test_exec_help_for_an_installed_reference_includes_program_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    module = package_root / "tools" / "review.agl"
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
    module = package_root / "tools" / "review.agl"
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
    module = package_root / "tools" / "lint.agl"
    module.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n\n'
        '[commands]\n"tools lint" = { program = "tools/lint::main", '
        'description = "Lint package inputs" }\n',
        encoding="utf-8",
    )
    module.write_text("program def main(level: text) -> unit = ()\n", encoding="utf-8")
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


def test_registered_command_argument_error_handles_no_description_or_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    module = package_root / "tools" / "lint.agl"
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


@pytest.mark.parametrize("reference", ["not-a-reference", "bad-name/main::main"])
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
    module = root / "tools" / "review.agl"
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
    updated = root / "tools" / "updated.agl"
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
    tools_module = tools_root / "tools" / "review.agl"
    bravo_module = bravo_root / "bravo" / "review.agl"
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
    entry = package_root / "tools" / "main.agl"
    helper = package_root / "tools" / "helper.agl"
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
