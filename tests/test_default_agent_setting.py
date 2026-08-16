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
from agm.commands import exec_program as exec_engine
from agm.config.context import ConfigContext
from tests._agl_helpers import agent_value, run_inline_command

_STDLIB = Path(__file__).resolve().parent.parent / "stdlib"


def _file_program(body: str) -> str:
    """Build an explicit file entry while leaving import headers at the root."""
    lines = body.splitlines()
    headers: list[str] = []
    while lines and (lines[0].startswith("import ") or lines[0].startswith("open import ")):
        headers.append(lines.pop(0))
    return "\n".join(
        (*headers, "program def main() -> unit =", *(f"  {line}" for line in lines), "")
    )


def _run(
    source: str,
    *,
    seed: dict[str, Value] | None = None,
    host_settings_policy: object | None = None,
) -> RunResult:
    result = run_inline_command(
        PipelineDriver(),
        source,
        roots=RootSet(roots=frozenset({_STDLIB})),
        builtin_host_settings=seed,
        host_settings_policy=host_settings_policy,
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
    assert result.bindings["initial"] == agent_value(
        "AgentClaude", model="sonnet", thinking="medium"
    )
    assert result.bindings["updated"] == agent_value("AgentCommand", command="command")


@pytest.mark.parametrize(
    ("write", "expected"),
    [
        ("std/config::log := true", (True, None)),
        ('std/config::log-file := Some("trace.jsonl")', (True, "trace.jsonl")),
    ],
)
def test_trace_setting_writes_reconfigure_trace(
    write: str, expected: tuple[bool, str | None]
) -> None:
    """Each trace-register write repoints the same live trace store."""
    from agm.agl.runtime.host_settings import HostSettingsPolicy

    trace_settings: list[tuple[bool, str | None]] = []

    def resolve_trace_path(enabled: bool, log_file: str | None) -> None:
        trace_settings.append((enabled, log_file))
        return None

    result = _run(
        f"import std/config\n{write}\n()\n",
        host_settings_policy=HostSettingsPolicy(resolve_trace_path=resolve_trace_path),
    )

    assert result.ok
    assert trace_settings == [(False, None), expected]


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
    program.write_text(_file_program("import std/config\nprint std/config::default-agent\n"))

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
        seed={"default-agent": agent_value("AgentCodex", model="o3", thinking="medium")},
    )

    assert result.ok
    assert result.bindings["seeded"] == agent_value("AgentCodex", model="o3", thinking="medium")
    assert result.bindings["written"] == agent_value(
        "AgentPi", provider="openai", model="gpt", thinking="high"
    )


@pytest.mark.parametrize(
    ("config_literal", "cli_literal", "source_literal", "expected"),
    (
        (
            'AgentCommand("config")',
            None,
            None,
            agent_value("AgentCommand", command="config"),
        ),
        (
            'AgentCommand("config")',
            'AgentCodex("o3", "medium")',
            None,
            agent_value("AgentCodex", model="o3", thinking="medium"),
        ),
        (
            'AgentCommand("config")',
            'AgentCodex("o3", "medium")',
            'AgentPi("openai", "gpt", "high")',
            agent_value("AgentPi", provider="openai", model="gpt", thinking="high"),
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
        _file_program(
            f"import std/config\n{source_write}let value = std/config::default-agent\nprint value\n"
        )
    )
    monkeypatch.setattr(
        exec_engine,
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


def test_exec_runner_config_seeds_default_agent_as_agent_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``[exec] runner`` is a bare host command, decoded into ``AgentCommand``.

    Unlike ``default-agent``, it is never rendered as AgL source (it carries
    quotes/backslashes/``%{`` verbatim), so it is decoded directly into a
    typed value rather than spliced in as an override.
    """
    home = tmp_path / "home"
    config_dir = home / ".agm"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('[exec]\nrunner = "claude"\n')
    program = tmp_path / "program.agl"
    program.write_text(_file_program("import std/config\nprint std/config::default-agent\n"))
    monkeypatch.setattr(
        exec_engine,
        "current_config_context",
        lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
    )

    assert (
        exec_command.run(ExecArgs(file=str(program), strict_json=None, no_log=True, log_file=None))
        is None
    )

    rendered = capsys.readouterr().out
    assert "AgentCommand" in rendered
    assert "claude" in rendered


@pytest.mark.parametrize(
    ("default_agent_literal", "runner", "expected"),
    (
        (None, "claude", agent_value("AgentCommand", command="claude")),
        (
            'AgentCommand("configured")',
            "claude",
            agent_value("AgentCommand", command="configured"),
        ),
    ),
)
def test_exec_default_agent_beats_runner_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    default_agent_literal: str | None,
    runner: str,
    expected: EnumValue,
) -> None:
    home = tmp_path / "home"
    config_dir = home / ".agm"
    config_dir.mkdir(parents=True)
    default_agent_line = (
        "" if default_agent_literal is None else f"default-agent = {default_agent_literal!r}\n"
    )
    (config_dir / "config.toml").write_text(f'[exec]\nrunner = "{runner}"\n{default_agent_line}')
    program = tmp_path / "program.agl"
    program.write_text(_file_program("import std/config\nprint std/config::default-agent\n"))
    monkeypatch.setattr(
        exec_engine,
        "current_config_context",
        lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
    )

    assert (
        exec_command.run(ExecArgs(file=str(program), strict_json=None, no_log=True, log_file=None))
        is None
    )

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
    program.write_text(_file_program('print "not-run"\n'))

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
    assert "--agent" in error


@pytest.mark.parametrize("toml_value", ['"AgentCommand("', "7"])
def test_exec_rejects_invalid_agent_literal_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    toml_value: str,
) -> None:
    home = tmp_path / "home"
    config_dir = home / ".agm"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(f"[exec]\ndefault-agent = {toml_value}\n")
    program = tmp_path / "program.agl"
    program.write_text(_file_program('print "not-run"\n'))
    monkeypatch.setattr(
        exec_engine,
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


def test_exec_rejects_blank_agent_literal_from_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A blank ``--agent`` is a host-shape error, diagnosed before any AgL parsing."""
    program = tmp_path / "program.agl"
    program.write_text(_file_program('print "not-run"\n'))

    with pytest.raises(SystemExit) as exc_info:
        exec_command.run(
            ExecArgs(file=str(program), strict_json=None, agent="", no_log=True, log_file=None)
        )

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "default-agent" in error
    assert "--agent" in error


def test_exec_rejects_malformed_exec_runner_before_any_module_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``[exec] runner`` is shell-split eagerly, before the program is even loaded.

    An unclosed quote is a host-configuration error (exit 1): the module
    graph is never loaded, so the program's own output never appears.
    """
    home = tmp_path / "home"
    config_dir = home / ".agm"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('[exec]\nrunner = "claude \'oops"\n')
    program = tmp_path / "program.agl"
    program.write_text(_file_program('print "not-run"\n'))
    monkeypatch.setattr(
        exec_engine,
        "current_config_context",
        lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
    )

    with pytest.raises(SystemExit) as exc_info:
        exec_command.run(ExecArgs(file=str(program), strict_json=None, no_log=True, log_file=None))

    assert exc_info.value.code == 1
    out, error = capsys.readouterr()
    assert out == ""
    assert "runner" in error


def test_exec_rejects_malformed_agent_command_literal_from_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A well-typed but unparseable ``AgentCommand`` literal exits 1 before any run.

    Unlike a blank/non-``Agent`` literal (rejected before AgL parsing even
    starts) or an ill-typed one (rejected by ``std/config``'s own
    typechecking), ``AgentCommand("nonexistent-bin -p 'oops")`` is a
    syntactically valid constant ``Agent`` expression -- its unclosed quote
    only breaks shell-splitting, which is checked when the interpreter
    materializes the winning ``default-agent`` value, before the program's
    first statement (its own ``print``, ahead of its first ``ask``) runs.
    """
    program = tmp_path / "program.agl"
    program.write_text(_file_program('print "before ask"\nask "hello"\n'))

    with pytest.raises(SystemExit) as exc_info:
        exec_command.run(
            ExecArgs(
                file=str(program),
                strict_json=None,
                agent='AgentCommand("nonexistent-bin -p \'oops")',
                no_log=True,
                log_file=None,
            )
        )

    assert exc_info.value.code == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "before ask" not in out
    assert "agent" in err.lower()


def test_exec_allows_well_formed_agent_command_literal_from_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A well-formed ``AgentCommand`` literal is not rejected, and the program runs."""
    program = tmp_path / "program.agl"
    program.write_text(_file_program('print "ran"\n'))

    assert (
        exec_command.run(
            ExecArgs(
                file=str(program),
                strict_json=None,
                agent='AgentCommand("echo hi")',
                no_log=True,
                log_file=None,
            )
        )
        is None
    )

    assert capsys.readouterr().out == "ran\n"


def test_source_write_of_malformed_agent_command_stays_a_runtime_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A runtime ``default-agent := AgentCommand(...)`` write is an AgL exception, not a host exit.

    Only the value materialized at interpreter construction is validated
    eagerly; a source write happens after construction, so a malformed
    command written at runtime surfaces as an ordinary uncaught AgL
    exception (exit 2) raised from the ``ask`` call site that dispatches it,
    exactly as before this change -- never a host-configuration exit.
    """
    program = tmp_path / "program.agl"
    program.write_text(
        _file_program(
            "import std/config\n"
            'print "before ask"\n'
            'std/config::default-agent := AgentCommand("nonexistent-bin -p \'oops")\n'
            'ask "hello"\n'
        )
    )

    with pytest.raises(SystemExit) as exc_info:
        exec_command.run(ExecArgs(file=str(program), strict_json=None, no_log=True, log_file=None))

    assert exc_info.value.code == 2
    out, err = capsys.readouterr()
    assert out == "before ask\n"
    assert "SessionError" in err
