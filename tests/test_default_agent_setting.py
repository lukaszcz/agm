"""Behavior tests for the typed ``std/config::default-agent`` engine setting."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
from agm.agl.pipeline import PipelineDriver, RunResult
from agm.agl.runtime.agents import agent_member_name
from agm.agl.semantics.values import BoolValue, RecordValue, TextValue, Value
from agm.cli_support.args import ExecArgs
from agm.commands import exec as exec_command
from agm.commands import exec_program as exec_engine
from agm.config.context import ConfigContext
from tests._agl_helpers import agent_value, agl_roots, prepare_inline_command, run_inline_command


def _file_program(body: str) -> str:
    """Build an explicit file entry while leaving import headers at the root."""
    lines = body.splitlines()
    headers: list[str] = []
    while lines and lines[0].startswith("import "):
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
        roots=agl_roots(),
        builtin_host_settings=seed,
        host_settings_policy=host_settings_policy,
    )
    assert isinstance(result, RunResult)
    return result


def _assert_agent_shape(actual: Value, is_variant: Value, expected: RecordValue) -> None:
    """Verify *actual* is the expected ``Agent`` variant with the expected payload.

    *is_variant* is an ``is`` member test run inside the program's own
    source (bound alongside *actual*): the running program loads real stdlib,
    so its own ``Agent`` enum carries that program's own nominal identity,
    distinct from the reserved-fallback identity *expected* (built by the
    shared ``agent_value`` test helper) carries. Only a cast evaluated inside
    that same program can compare identity correctly; fields compare directly
    since they carry no identity of their own.
    """
    assert is_variant == BoolValue(True)
    assert isinstance(actual, RecordValue)
    assert actual.fields == expected.fields


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
        "let initial-is-claude = initial is Agent::AgentClaude\n"
        'std/config::default-agent := AgentCommand("command")\n'
        "let updated = std/config::default-agent\n"
        "let updated-is-command = updated is Agent::AgentCommand\n"
        "updated\n"
    )

    assert result.ok
    _assert_agent_shape(
        result.bindings["initial"],
        result.bindings["initial-is-claude"],
        agent_value("AgentClaude", model="sonnet", thinking="medium"),
    )
    _assert_agent_shape(
        result.bindings["updated"],
        result.bindings["updated-is-command"],
        agent_value("AgentCommand", command="command"),
    )


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
        "let seeded-is-codex = seeded is Agent::AgentCodex\n"
        'std/config::default-agent := AgentPi("openai", "gpt", "high")\n'
        "let written = std/config::default-agent\n"
        "let written-is-pi = written is Agent::AgentPi\n"
        "written\n",
        seed={"default-agent": agent_value("AgentCodex", model="o3", thinking="medium")},
    )

    assert result.ok
    _assert_agent_shape(
        result.bindings["seeded"],
        result.bindings["seeded-is-codex"],
        agent_value("AgentCodex", model="o3", thinking="medium"),
    )
    _assert_agent_shape(
        result.bindings["written"],
        result.bindings["written-is-pi"],
        agent_value("AgentPi", provider="openai", model="gpt", thinking="high"),
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
    expected: RecordValue,
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
                default_agent=cli_literal,
                no_log=True,
                log_file=None,
            )
        )
        is None
    )

    # ``run`` retains top-level bindings only internally; the observable output
    # confirms the selected constructor and each expected field.
    rendered = capsys.readouterr().out
    assert agent_member_name(expected, NO_BUILTIN_DECLARATIONS) in rendered
    for field in expected.fields.values():
        assert isinstance(field, TextValue)
        assert field.value in rendered


def test_program_table_default_agent_beats_exec_default_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A qualified program table's ``default-agent`` outranks ``[exec]``."""
    home = tmp_path / "home"
    config_dir = home / ".agm"
    config_dir.mkdir(parents=True)
    exec_literal = 'AgentCommand("exec-level")'
    program_literal = 'AgentCommand("program-level")'
    config_dir.joinpath("config.toml").write_text(
        f"[exec]\ndefault-agent = {exec_literal!r}\n\n"
        f"[prog.main]\ndefault-agent = {program_literal!r}\n"
    )
    program = tmp_path / "prog.agl"
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
    assert "program-level" in rendered
    assert "exec-level" not in rendered


def test_cli_default_agent_beats_program_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--default-agent`` outranks a qualified program table's own value."""
    home = tmp_path / "home"
    config_dir = home / ".agm"
    config_dir.mkdir(parents=True)
    program_literal = 'AgentCommand("program-level")'
    config_dir.joinpath("config.toml").write_text(
        f"[prog.main]\ndefault-agent = {program_literal!r}\n"
    )
    program = tmp_path / "prog.agl"
    program.write_text(_file_program("import std/config\nprint std/config::default-agent\n"))
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
                default_agent='AgentCommand("cli-level")',
                no_log=True,
                log_file=None,
            )
        )
        is None
    )

    rendered = capsys.readouterr().out
    assert "cli-level" in rendered
    assert "program-level" not in rendered


def test_exec_config_runner_key_leaves_the_default_agent_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``[exec] runner`` is not a setting: even a malformed one is inert."""
    home = tmp_path / "home"
    config_dir = home / ".agm"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('[exec]\nrunner = "claude \'oops"\n')
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
    assert "AgentClaude" in rendered
    assert "AgentCommand" not in rendered


@pytest.mark.parametrize("value", ["true", "worker --flag"])
def test_exec_treats_non_agent_syntax_from_cli_as_a_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    program = tmp_path / "program.agl"
    program.write_text(_file_program('print "ran"\n'))

    assert (
        exec_command.run(
            ExecArgs(
                file=str(program),
                strict_json=None,
                default_agent=value,
                no_log=True,
                log_file=None,
            )
        )
        is None
    )
    assert capsys.readouterr().out == "ran\n"


@pytest.mark.parametrize("value", ["AgentCommand(", 'AgentCommand("x") + "y"'])
def test_exec_rejects_text_that_opens_a_member_call_but_fails_to_read_from_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    """Text naming a real ``Agent`` member commits to constructor-call syntax:
    a read failure inside it is a host error, not a verbatim command."""
    program = tmp_path / "program.agl"
    program.write_text(_file_program('print "not-run"\n'))

    with pytest.raises(SystemExit) as exc_info:
        exec_command.run(
            ExecArgs(
                file=str(program),
                strict_json=None,
                default_agent=value,
                no_log=True,
                log_file=None,
            )
        )

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "--default-agent" in error


@pytest.mark.parametrize("toml_value", ["7"])
def test_exec_rejects_non_string_agent_value_from_config(
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


def test_exec_treats_non_agent_syntax_from_config_as_a_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "home"
    config_dir = home / ".agm"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('[exec]\ndefault-agent = "not an agent"\n')
    program = tmp_path / "program.agl"
    program.write_text(_file_program('print "ran"\n'))
    monkeypatch.setattr(
        exec_engine,
        "current_config_context",
        lambda: ConfigContext(home=home, proj_dir=None, cwd=tmp_path),
    )

    assert (
        exec_command.run(ExecArgs(file=str(program), strict_json=None, no_log=True, log_file=None))
        is None
    )
    assert capsys.readouterr().out == "ran\n"


def test_exec_rejects_blank_agent_literal_from_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A blank ``--default-agent`` is a host-shape error, diagnosed before any AgL parsing."""
    program = tmp_path / "program.agl"
    program.write_text(_file_program('print "not-run"\n'))

    with pytest.raises(SystemExit) as exc_info:
        exec_command.run(
            ExecArgs(
                file=str(program), strict_json=None, default_agent="", no_log=True, log_file=None
            )
        )

    assert exc_info.value.code == 1
    error = capsys.readouterr().err
    assert "default-agent" in error
    assert "--default-agent" in error


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
                default_agent='AgentCommand("nonexistent-bin -p \'oops")',
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
                default_agent='AgentCommand("echo hi")',
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
    exception (exit 2) raised from the ``ask`` call site that dispatches
    it -- never a host-configuration exit.
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


class TestMalformedAgentCommandAtConstruction:
    """``AgentCommand("...")`` is a valid constant ``Agent`` value even when its
    command text does not shell-split, so preparing the program accepts it
    cleanly; the malformed text is only caught eagerly when the interpreter
    is constructed, before the program's first statement runs.
    """

    def test_seeded_malformed_command_text_is_a_pre_execution_diagnostic(self) -> None:
        driver = PipelineDriver()
        prepared = prepare_inline_command("()", entry_path=None, roots=agl_roots())
        result = driver.run_prepared(
            prepared,
            builtin_host_settings={
                "default-agent": agent_value("AgentCommand", command="nonexistent-bin -p 'oops")
            },
        )
        assert not result.ok
        assert result.diagnostics
        assert result.error is None


def test_restamp_engine_setting_ignores_non_enum_backed_keys() -> None:
    """A boolean-kind or unrecognized key carries no host-enum identity to restamp.

    ``log``/``strict-json`` are boolean-kind and an unrecognized name is not a
    key at all, so :func:`restamp_engine_setting` leaves such a value untouched
    regardless of the ``from``/``to`` tables given.
    """
    from agm.agl.runtime.engine_config import restamp_engine_setting

    stray = agent_value("AgentCommand", command="unused")
    for key in ("log", "strict-json", "not-an-engine-key"):
        assert (
            restamp_engine_setting(
                key, stray, from_table=NO_BUILTIN_DECLARATIONS, to_table=NO_BUILTIN_DECLARATIONS
            )
            is stray
        )
