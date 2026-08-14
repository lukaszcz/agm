"""Tests for installed package command dispatch and references."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import semver
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
from agm.agl.runtime.types import ParamDeclInfo
from agm.agl.semantics.types import BoolType, TextType
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


def invoke(runner: CliRunner, argv: list[str], *, env: dict[str, str] | None = None) -> Result:
    return runner.invoke(
        get_command(cli.app), argv, prog_name="agm", catch_exceptions=False, env=env
    )


def test_unexpected_command_resolution_errors_are_not_treated_as_registered_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agm.cli_dispatch as dispatch

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
    import agm.cli_dispatch as dispatch
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
    import agm.cli_dispatch as dispatch
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
        lambda program, param_tokens, *, package, command_path: calls.append(
            (program, param_tokens, package, command_path)
        ),
    )

    result = invoke(CliRunner(), ["tools", "lint", "--level", "strict"])

    assert result.exit_code == 0
    assert calls == [("tools/lint::main", ["--level", "strict"], "tools", "tools lint")]


def test_registered_command_treats_only_standalone_dry_run_as_global(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_dispatch as dispatch
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
        lambda _program, param_tokens, **_kwargs: calls.append((param_tokens, dry_run.enabled())),
    )

    value_result = invoke(CliRunner(), ["tools", "lint", "--level=--dry-run"])
    flag_result = invoke(CliRunner(), ["tools", "lint", "--level", "strict", "--dry-run"])

    assert value_result.exit_code == 0
    assert flag_result.exit_code == 0
    assert calls == [(["--level=--dry-run"], False), (["--level", "strict"], True)]


def test_registered_command_help_does_not_dispatch_program(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_dispatch as dispatch
    import agm.commands.exec_program as exec_program

    context = ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)
    index = ActivationIndex(
        commands={
            "tools lint": CommandRegistration("tools", "tools/lint::main", "Lint package inputs")
        }
    )
    monkeypatch.setattr(dispatch, "current_config_context", lambda: context)
    monkeypatch.setattr(dispatch, "load_command_index", lambda **_: index)
    params = (
        ParamDeclInfo("level", TextType(), False, 1, 1),
        ParamDeclInfo("verbose", BoolType(), False, 2, 1),
        ParamDeclInfo("message", TextType(), False, 3, 1),
    )
    monkeypatch.setattr(exec_program, "registered_program_params", lambda *_a, **_k: params)
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


def test_help_command_renders_registered_command_help(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_dispatch as dispatch

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


def test_registered_command_help_degrades_when_param_discovery_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agm.cli_dispatch as dispatch
    import agm.commands.exec_program as exec_program

    monkeypatch.setattr(
        exec_program,
        "registered_program_param_flags",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    text = dispatch.registered_command_help(
        "tools lint", CommandRegistration("tools", "tools/lint::main")
    )

    assert "Run the registered AgL program." in text
    assert "Program parameters:" not in text

    monkeypatch.setattr(exec_program, "registered_program_param_flags", lambda *_args: ("--level",))

    text = dispatch.registered_command_help(
        "tools lint", CommandRegistration("tools", "tools/lint::main")
    )

    assert "Program parameters:\n  --level" in text


def test_registered_command_help_returns_false_when_index_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_dispatch as dispatch

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


def test_registered_param_flags_reject_invalid_or_mismatched_references() -> None:
    import agm.commands.exec_program as exec_program

    assert exec_program.registered_program_param_flags("not-a-reference", "tools") == ()
    assert exec_program.registered_program_param_flags("tools/bad-name::main", "tools") == ()
    assert exec_program.registered_program_param_flags("other/lint::main", "tools") == ()


def test_registered_param_flags_degrade_when_selection_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.commands.exec_program as exec_program

    context = ConfigContext(tmp_path / "home", None, tmp_path)
    monkeypatch.setattr(
        exec_program,
        "select_active_packages",
        lambda **_: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    assert (
        exec_program.registered_program_param_flags("tools/lint::main", "tools", context=context)
        == ()
    )


def test_unknown_command_without_registered_entry_keeps_click_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_dispatch as dispatch

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
    import agm.cli_dispatch as dispatch

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
    import agm.cli_dispatch as dispatch

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
    import agm.commands.exec as exec_command
    import agm.commands.exec_program as exec_program

    root = tmp_path / "tools"
    module = root / "tools" / "review.agl"
    module.parent.mkdir(parents=True)
    module.write_text("program def main() -> unit = ()\n", encoding="utf-8")
    package = PackageInfo(root, PackageManifest("tools", semver.Version.parse("1.0.0")))
    context = ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path)
    monkeypatch.setattr(exec_program, "current_config_context", lambda: context)
    monkeypatch.setattr(exec_program, "select_active_packages", lambda **_: (package,))
    calls: list[ExecArgs] = []

    def fake_run(args: ExecArgs, **_: object) -> None:
        calls.append(args)

    monkeypatch.setattr(exec_program, "run", fake_run)

    args = ExecArgs(
        file="tools/review::main",
        strict_json=True,
        no_log=True,
        log_file="trace.jsonl",
        param_tokens=["--subject", "changes"],
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
    (package_root / "tools" / "settings.agl").write_text(
        'param level: text = "default"\n', encoding="utf-8"
    )
    module.write_text(
        "import tools/settings\nparam level: text\nprogram def main() -> unit = print level\n",
        encoding="utf-8",
    )
    write_record(package_root)
    write_activation_index(
        ActivationIndex({"tools": ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    monkeypatch.setenv("HOME", str(home))

    result = invoke(CliRunner(), ["exec", "tools/review::main", "--tools/review::level", "set"])

    assert result.exit_code == 0
    assert result.stdout == "set\n"


def test_exec_help_for_an_installed_reference_includes_program_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    package_root = home / ".agm" / "packages" / "tools" / "1.0.0"
    module = package_root / "tools" / "review.agl"
    module.parent.mkdir(parents=True)
    (package_root / "package.toml").write_text(
        '[package]\nname = "tools"\nversion = "1.0.0"\n', encoding="utf-8"
    )
    (package_root / "tools" / "settings.agl").write_text(
        'param level: text = "default"\n', encoding="utf-8"
    )
    module.write_text(
        "import tools/settings\nparam level: text\nprogram def main() -> unit = ()\n",
        encoding="utf-8",
    )
    write_record(package_root)
    write_activation_index(
        ActivationIndex({"tools": ActivePackage(semver.Version.parse("1.0.0"))}), home=home
    )
    monkeypatch.setenv("HOME", str(home))

    result = invoke(CliRunner(), ["exec", "tools/review::main", "--help"])

    assert result.exit_code == 0
    assert "--tools/review::level" in result.output


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


def test_registered_param_errors_show_registered_command_usage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import agm.commands.exec_program as exec_program

    source = tmp_path / "main.agl"
    source.write_text("param level: text\nprogram def main() -> unit = ()\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        exec_program.run(
            ExecArgs(
                file=str(source),
                param_tokens=["--unknown"],
                strict_json=None,
                no_log=False,
                log_file=None,
                no_stdlib=True,
            ),
            usage_command_path="tools lint",
            usage_description="Lint package inputs",
        )

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "usage: agm tools lint" in error
    assert "Lint package inputs" in error
    assert "--level" in error
    assert "agm exec" not in error


def test_registered_param_error_usage_handles_no_description_or_params(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import agm.commands.exec_program as exec_program

    source = tmp_path / "main.agl"
    source.write_text("program def main() -> unit = ()\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        exec_program.run(
            ExecArgs(
                file=str(source),
                param_tokens=["--unknown"],
                strict_json=None,
                no_log=False,
                log_file=None,
                no_stdlib=True,
            ),
            usage_command_path="tools lint",
        )

    assert "usage: agm tools lint" in capsys.readouterr().err


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
    import agm.commands.exec_program as exec_program

    monkeypatch.setattr(
        exec_program,
        "current_config_context",
        lambda: ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path),
    )
    monkeypatch.setattr(
        exec_program,
        "select_active_packages",
        lambda **_: (_ for _ in ()).throw(ValueError("broken selection")),
    )

    with pytest.raises(SystemExit) as exc_info:
        exec_program.run_registered("tools/review::main", [])

    assert exc_info.value.code == 1


def test_registered_dispatch_rejects_a_stale_cached_program(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    monkeypatch.setattr(exec_program, "select_active_packages", lambda **_: (package,))

    with pytest.raises(SystemExit) as exc_info:
        exec_program.run_registered(
            "tools/review::other", [], package="tools", command_path="review"
        )

    assert exc_info.value.code == 1


def test_editable_registered_dispatch_uses_the_live_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    monkeypatch.setattr(exec_program, "select_active_packages", lambda **_: (package,))
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
                param_tokens=[],
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
    monkeypatch.setattr(exec_program, "select_active_packages", lambda **_: (package,))
    monkeypatch.setattr(exec_program, "load_activation_index", lambda **_: index)

    with pytest.raises(SystemExit) as exc_info:
        exec_program.run_registered("tools/main::main", [], package="tools", command_path="live")

    assert exc_info.value.code == 1


def test_registered_dispatch_rejects_a_program_owned_by_another_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    monkeypatch.setattr(exec_program, "select_active_packages", lambda **_: (tools, bravo))

    with pytest.raises(SystemExit) as exc_info:
        exec_program.run_registered(
            "tools/review::main", [], package="tools", command_path="review"
        )

    assert exc_info.value.code == 1


def test_malformed_activation_index_fallback_is_a_clean_cli_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.cli_dispatch as dispatch

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
    import agm.commands.exec_program as exec_program

    package = PackageInfo(
        tmp_path / "tools", PackageManifest("tools", semver.Version.parse("1.0.0"))
    )
    monkeypatch.setattr(
        exec_program,
        "current_config_context",
        lambda: ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path),
    )
    monkeypatch.setattr(exec_program, "select_active_packages", lambda **_: (package,))

    with pytest.raises(SystemExit) as exc_info:
        exec_program.run_registered("tools/review::main", [])

    assert exc_info.value.code == 1


def test_exec_unknown_installed_reference_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import agm.commands.exec as exec_command
    import agm.commands.exec_program as exec_program

    monkeypatch.setattr(
        exec_program,
        "current_config_context",
        lambda: ConfigContext(home=tmp_path / "home", proj_dir=None, cwd=tmp_path),
    )
    monkeypatch.setattr(exec_program, "select_active_packages", lambda **_: ())

    with pytest.raises(SystemExit) as exc_info:
        exec_command.run(
            ExecArgs(file="missing/main::main", strict_json=None, no_log=False, log_file=None)
        )

    assert exc_info.value.code == 1
