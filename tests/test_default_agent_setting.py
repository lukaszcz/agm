"""Behavior tests for the typed ``std/config::default-agent`` engine setting."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver, RunResult
from agm.agl.semantics.values import EnumValue, TextValue, Value
from agm.cli_support.args import ExecArgs
from agm.commands import exec as exec_command
from agm.config.context import ConfigContext

_STDLIB = Path(__file__).resolve().parent.parent / "stdlib"


def _agent(variant: str, **fields: str) -> EnumValue:
    """Return the expected runtime representation of an ``Agent`` value."""
    from agm.agl.ir.ids import NominalId
    from agm.agl.modules.ids import STD_CORE_ID

    return EnumValue(
        nominal=NominalId(STD_CORE_ID, "Agent"),
        display_name="Agent",
        variant=variant,
        fields={name: TextValue(value) for name, value in fields.items()},
    )


def _run(
    source: str,
    *,
    seed: dict[str, Value] | None = None,
    host_settings_policy: object | None = None,
) -> RunResult:
    driver = PipelineDriver()
    prepared = driver.prepare_program(
        source, entry_path=None, roots=RootSet(roots=frozenset({_STDLIB}))
    )
    result = driver.run_prepared(
        prepared, builtin_host_settings=seed, host_settings_policy=host_settings_policy
    )
    assert isinstance(result, RunResult)
    return result


def test_engine_key_uses_the_agent_nominal_type() -> None:
    from agm.agl.semantics.engine_keys import get_engine_key_type

    assert repr(get_engine_key_type("default-agent")) == "Agent"


def test_non_agl_command_does_not_import_agl() -> None:
    script = (
        "import sys\n"
        "from click.testing import CliRunner\n"
        "from typer.main import get_command\n"
        "import agm.cli as cli\n"
        "result = CliRunner().invoke(get_command(cli.app), ['config', 'env'])\n"
        "assert result.exit_code == 0, result.output\n"
        "assert not any(name == 'agm.agl' or name.startswith('agm.agl.') for name in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_default_agent_initializer_and_qualified_write_are_visible() -> None:
    result = _run(
        "import std/config\n"
        "let initial = std/config::default-agent\n"
        'std/config::default-agent := AgentCommand("command")\n'
        "let updated = std/config::default-agent\n"
        "updated\n"
    )

    assert result.ok
    assert result.bindings["initial"] == _agent("AgentClaude", model="sonnet", thinking="medium")
    assert result.bindings["updated"] == _agent("AgentCommand", command="command")


def test_default_agent_write_does_not_reconfigure_host_services() -> None:
    from agm.agl.runtime.host_settings import HostSettingsPolicy

    trace_settings: list[tuple[bool, str | None]] = []

    def resolve_trace_path(enabled: bool, log_file: str | None) -> None:
        trace_settings.append((enabled, log_file))
        return None

    result = _run(
        'import std/config\nstd/config::default-agent := AgentCommand("command")\n()\n',
        host_settings_policy=HostSettingsPolicy(resolve_trace_path=resolve_trace_path),
    )

    assert result.ok
    # The register-only write does not reconfigure tracing.
    assert trace_settings == [(False, None)]


def test_exec_uses_stdlib_default_agent_when_no_host_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    program = tmp_path / "program.agl"
    program.write_text("import std/config\nprint std/config::default-agent\n")

    assert (
        exec_command.run(
            ExecArgs(
                file=str(program),
                strict_json=None,
                no_log=True,
                log_file=None,
            )
        )
        is None
    )

    assert "AgentClaude" in capsys.readouterr().out


def test_host_seed_overrides_initializer_until_source_write() -> None:
    result = _run(
        "import std/config\n"
        "let seeded = std/config::default-agent\n"
        'std/config::default-agent := AgentPi("openai", "gpt", "high")\n'
        "let written = std/config::default-agent\n"
        "written\n",
        seed={"default-agent": _agent("AgentCodex", model="o3", thinking="medium")},
    )

    assert result.ok
    assert result.bindings["seeded"] == _agent("AgentCodex", model="o3", thinking="medium")
    assert result.bindings["written"] == _agent(
        "AgentPi", provider="openai", model="gpt", thinking="high"
    )


@pytest.mark.parametrize(
    ("config_literal", "cli_literal", "source_literal", "expected"),
    (
        (
            'AgentCommand("config")',
            None,
            None,
            _agent("AgentCommand", command="config"),
        ),
        (
            'AgentCommand("config")',
            'AgentCodex("o3", "medium")',
            None,
            _agent("AgentCodex", model="o3", thinking="medium"),
        ),
        (
            'AgentCommand("config")',
            'AgentCodex("o3", "medium")',
            'AgentPi("openai", "gpt", "high")',
            _agent("AgentPi", provider="openai", model="gpt", thinking="high"),
        ),
    ),
)
def test_exec_agent_source_cli_and_config_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config_literal: str,
    cli_literal: str | None,
    source_literal: str | None,
    expected: EnumValue,
) -> None:
    home = tmp_path / "home"
    config_dir = home / ".agm"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(f"[exec]\ndefault-agent = {config_literal!r}\n")
    source_write = (
        "" if source_literal is None else f"std/config::default-agent := {source_literal}\n"
    )
    program = tmp_path / "program.agl"
    program.write_text(
        f"import std/config\n{source_write}let value = std/config::default-agent\nprint value\n"
    )
    monkeypatch.setattr(
        exec_command,
        "current_config_context",
        lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
    )

    assert (
        exec_command.run(
            ExecArgs(
                file=str(program),
                strict_json=None,
                agent=cli_literal,
                no_log=True,
                log_file=None,
            )
        )
        is None
    )

    # ``run`` retains top-level bindings only internally; the observable output
    # confirms the selected constructor and each expected field.
    rendered = capsys.readouterr().out
    assert expected.variant in rendered
    for field in expected.fields.values():
        assert isinstance(field, TextValue)
        assert field.value in rendered


@pytest.mark.parametrize("literal", ["AgentCommand(", "true", 'AgentCommand("x") + "y"'])
def test_exec_rejects_invalid_agent_literal_from_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], literal: str
) -> None:
    program = tmp_path / "program.agl"
    program.write_text('print "not-run"\n')

    with pytest.raises(SystemExit) as exc_info:
        exec_command.run(
            ExecArgs(
                file=str(program),
                strict_json=None,
                agent=literal,
                no_log=True,
                log_file=None,
            )
        )

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "default-agent" in error
    assert literal in error


@pytest.mark.parametrize(
    ("toml_value", "literal"),
    [('"not an agent"', "not an agent"), ('""', "''"), ("7", "7")],
)
def test_exec_rejects_invalid_agent_literal_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    toml_value: str,
    literal: str,
) -> None:
    home = tmp_path / "home"
    config_dir = home / ".agm"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(f"[exec]\ndefault-agent = {toml_value}\n")
    program = tmp_path / "program.agl"
    program.write_text('print "not-run"\n')
    monkeypatch.setattr(
        exec_command,
        "current_config_context",
        lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
    )

    with pytest.raises(SystemExit) as exc_info:
        exec_command.run(
            ExecArgs(
                file=str(program),
                strict_json=None,
                no_log=True,
                log_file=None,
            )
        )

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "default-agent" in error
    assert literal in error
