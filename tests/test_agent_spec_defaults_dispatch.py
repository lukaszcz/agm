"""``agm exec``, ``agm repl`` and registered commands apply the configured agent defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner, Result
from typer.main import get_command

import agm.cli as cli
from agm.agl.repl import ReplSession
from agm.cli_support.args import ReplArgs
from agm.commands import exec_program as exec_engine
from agm.commands import repl as repl_command
from agm.config.context import ConfigContext
from agm.packages.layout import MODULE_TREE_DIRNAME
from tests._agl_helpers import write_file_program
from tests._package_helpers import install_directory
from tests.conftest import FakeAgentTransport
from tests.test_agent_rpc import RpcStub

_AGM_COMMAND = get_command(cli.app)

_OPUS_HIGH = '[agent.claude]\neffort = "low"\n[agent.claude.opus]\neffort = "high"\n'


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run every command from *tmp_path* with a HOME of its own and no project."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("AGM_PROJECT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    return home


def _write_config(root: Path, body: str) -> None:
    (root / ".agm").mkdir(parents=True, exist_ok=True)
    (root / ".agm" / "config.toml").write_text(body)


def _invoke(argv: list[str]) -> Result:
    return CliRunner().invoke(_AGM_COMMAND, argv, prog_name="agm", catch_exceptions=False)


def _exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    *,
    home_config: str | None = None,
    project_config: str | None = None,
) -> Result:
    home = tmp_path / "home"
    if home_config is not None:
        _write_config(home, home_config)
    proj_dir = None
    if project_config is not None:
        proj_dir = tmp_path / "proj"
        (proj_dir / "config").mkdir(parents=True)
        (proj_dir / "config" / "config.toml").write_text(project_config)
    monkeypatch.setattr(
        exec_engine,
        "current_config_context",
        lambda: ConfigContext(home=home, proj_dir=proj_dir, cwd=tmp_path),
    )
    program = tmp_path / "program.agl"
    write_file_program(program, source)
    return _invoke(["exec", "--no-trace", str(program)])


def _flag_values(argv: list[str], flag: str) -> list[str]:
    return [argv[index + 1] for index, arg in enumerate(argv) if arg == flag]


_AGENT_ASK = (
    'let answer: text = AgentClaude("opus", "").ask("hi", sandbox = AgentSandbox::Disabled)\n'
    "print answer\n"
)
_SESSION_ASK = (
    'let session = Session::open(AgentClaude("opus", ""), sandbox = AgentSandbox::Disabled)\n'
    'let answer: text = session.ask("hi")\n'
    "print answer\n"
)
_FREE_ASK = (
    "import std/config\n"
    'std/config::default-agent := AgentClaude("opus", "")\n'
    "std/config::default-sandbox := AgentSandbox::Disabled\n"
    'let answer: text = ask("hi")\n'
    "print answer\n"
)


@pytest.mark.parametrize(
    "source", [_AGENT_ASK, _SESSION_ASK, _FREE_ASK], ids=["agent-ask", "session", "free-ask"]
)
@pytest.mark.parametrize("layer", ["home", "project"])
def test_exec_applies_the_configured_effort_to_an_agent_without_thinking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_agent_transport: FakeAgentTransport,
    source: str,
    layer: str,
) -> None:
    fake_agent_transport.queue(fake_agent_transport.success("done"))
    config = {f"{layer}_config": _OPUS_HIGH}

    result = _exec(tmp_path, monkeypatch, source, **config)

    assert result.exit_code == 0, result.output
    assert result.output == "done\n"
    [(_prompt, argv)] = fake_agent_transport.calls
    assert argv[0] == "claude"
    assert _flag_values(argv, "--effort") == ["high"]


def test_exec_applies_the_configured_effort_to_codex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_agent_transport: FakeAgentTransport
) -> None:
    fake_agent_transport.queue(fake_agent_transport.success("done"))
    source = _AGENT_ASK.replace('AgentClaude("opus", "")', 'AgentCodex("gpt-5.1-codex", "")')

    result = _exec(
        tmp_path,
        monkeypatch,
        source,
        home_config='[agent.codex."gpt-5.1-codex"]\neffort = "xhigh"\n',
    )

    assert result.exit_code == 0, result.output
    [(_prompt, argv)] = fake_agent_transport.calls
    assert argv[:2] == ["codex", "exec"]
    assert _flag_values(argv, "-c") == ["model_reasoning_effort=xhigh"]


def test_exec_applies_the_configured_effort_to_a_pi_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_agent_transport: FakeAgentTransport
) -> None:
    fake_agent_transport.queue(fake_agent_transport.success("done"))
    source = (
        'let session = Session::open(AgentPi("anthropic", "sonnet", ""), '
        "transport = Some(SessionTransport::Cli), sandbox = AgentSandbox::Disabled)\n"
        'let answer: text = session.ask("hi")\n'
        "print answer\n"
    )

    result = _exec(
        tmp_path,
        monkeypatch,
        source,
        home_config='[agent.pi]\neffort = "low"\n[agent.pi.anthropic]\neffort = "medium"\n',
    )

    assert result.exit_code == 0, result.output
    [(_prompt, argv)] = fake_agent_transport.calls
    assert argv[0] == "pi"
    assert _flag_values(argv, "--thinking") == ["medium"]


def test_every_retry_attempt_carries_the_configured_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_agent_transport: FakeAgentTransport
) -> None:
    fake_agent_transport.queue(
        fake_agent_transport.success("not a number"), fake_agent_transport.success("7")
    )
    source = (
        'let answer: int = AgentClaude("opus", "").ask("hi", '
        "on-parse-error = ParsePolicy::Retry(n = 2), sandbox = AgentSandbox::Disabled)\n"
        "print answer\n"
    )

    result = _exec(tmp_path, monkeypatch, source, home_config=_OPUS_HIGH)

    assert result.exit_code == 0, result.output
    assert result.output == "7\n"
    assert len(fake_agent_transport.calls) == 2
    for _prompt, argv in fake_agent_transport.calls:
        assert _flag_values(argv, "--effort") == ["high"]


_BUILTIN_DEFAULT_ASK = (
    "import std/config\n"
    "std/config::default-sandbox := AgentSandbox::Disabled\n"
    'let answer: text = ask("hi")\n'
    "print answer\n"
)


_OPUS_DEFAULT = '[agent.claude]\nmodel = "opus"\n[agent.claude.opus]\neffort = "high"\n'


@pytest.mark.parametrize(
    ("config", "model", "effort"),
    [
        (_OPUS_DEFAULT, ["opus"], ["high"]),
        ('[agent.claude]\neffort = "low"\n', [], ["low"]),
        (None, [], []),
    ],
    ids=["model-and-effort", "effort-only", "unconfigured"],
)
def test_a_plain_ask_on_the_builtin_default_agent_takes_the_configured_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_agent_transport: FakeAgentTransport,
    config: str | None,
    model: list[str],
    effort: list[str],
) -> None:
    fake_agent_transport.queue(fake_agent_transport.success("done"))

    result = _exec(tmp_path, monkeypatch, _BUILTIN_DEFAULT_ASK, home_config=config)

    assert result.exit_code == 0, result.output
    [(_prompt, argv)] = fake_agent_transport.calls
    assert argv[0] == "claude"
    assert _flag_values(argv, "--model") == model
    assert _flag_values(argv, "--effort") == effort


def _ask_with(agent: str) -> str:
    return _AGENT_ASK.replace('AgentClaude("opus", "")', agent)


@pytest.mark.parametrize("config", [_OPUS_DEFAULT, None], ids=["configured", "unconfigured"])
def test_exec_applies_the_configured_model_and_its_effort_to_claude(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_agent_transport: FakeAgentTransport,
    config: str | None,
) -> None:
    fake_agent_transport.queue(fake_agent_transport.success("done"))

    result = _exec(tmp_path, monkeypatch, _ask_with("AgentClaude()"), home_config=config)

    assert result.exit_code == 0, result.output
    [(_prompt, argv)] = fake_agent_transport.calls
    expected = ["--model", "opus", "--effort", "high"] if config else []
    assert argv == ["claude", "-p", *expected]


def test_exec_applies_the_configured_model_to_codex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_agent_transport: FakeAgentTransport
) -> None:
    fake_agent_transport.queue(fake_agent_transport.success("done"))
    config = '[agent.codex]\nmodel = "gpt-5.1-codex"\neffort = "low"\n'

    result = _exec(
        tmp_path, monkeypatch, _ask_with('AgentCodex(thinking = "high")'), home_config=config
    )

    assert result.exit_code == 0, result.output
    [(_prompt, argv)] = fake_agent_transport.calls
    assert argv == [
        "codex",
        "exec",
        "--model",
        "gpt-5.1-codex",
        "-c",
        "model_reasoning_effort=high",
        "-",
    ]


@pytest.mark.parametrize(
    ("agent", "expected"),
    [
        ("AgentPi()", ["--provider", "anthropic", "--model", "sonnet", "--thinking", "high"]),
        (
            'AgentPi(provider = "openai")',
            ["--provider", "openai", "--model", "fallback", "--thinking", "low"],
        ),
    ],
)
def test_exec_applies_the_configured_pi_provider_model_and_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_agent_transport: FakeAgentTransport,
    agent: str,
    expected: list[str],
) -> None:
    fake_agent_transport.queue(fake_agent_transport.success("done"))
    config = (
        '[agent.pi]\nprovider = "anthropic"\nmodel = "fallback"\neffort = "low"\n'
        '[agent.pi.anthropic]\nmodel = "sonnet"\n'
        '[agent.pi.anthropic.sonnet]\neffort = "high"\n'
    )

    source = (
        f"let session = Session::open({agent}, "
        "transport = Some(SessionTransport::Cli), sandbox = AgentSandbox::Disabled)\n"
        'let answer: text = session.ask("hi")\n'
        "print answer\n"
    )

    result = _exec(tmp_path, monkeypatch, source, project_config=config)

    assert result.exit_code == 0, result.output
    [(_prompt, argv)] = fake_agent_transport.calls
    assert argv[:2] == ["pi", "-p"]
    assert argv[-len(expected) :] == expected


def test_exec_applies_the_configured_pi_defaults_to_a_default_rpc_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = RpcStub(tmp_path, monkeypatch)
    config = (
        '[agent.pi]\nprovider = "anthropic"\n[agent.pi.anthropic]\nmodel = "sonnet"\n'
        '[agent.pi.anthropic.sonnet]\neffort = "high"\n'
    )
    source = (
        "let session = Session::open(AgentPi(), sandbox = AgentSandbox::Disabled)\n"
        'let answer: text = session.ask("hi")\n'
        "print answer\n"
    )

    result = _exec(tmp_path, monkeypatch, source, home_config=config)

    assert result.exit_code == 0, result.output
    assert result.output == "answer\n"
    [start] = stub.records("starts.jsonl")
    argv = start["argv"]
    assert isinstance(argv, list)
    assert argv[:2] == ["--mode", "rpc"]
    assert argv[-6:] == ["--provider", "anthropic", "--model", "sonnet", "--thinking", "high"]


def test_exec_without_configured_effort_passes_no_effort_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_agent_transport: FakeAgentTransport
) -> None:
    fake_agent_transport.queue(fake_agent_transport.success("done"))

    result = _exec(tmp_path, monkeypatch, _AGENT_ASK, home_config='[agent.codex]\neffort = "x"\n')

    assert result.exit_code == 0, result.output
    [(_prompt, argv)] = fake_agent_transport.calls
    assert _flag_values(argv, "--effort") == []


def test_an_explicit_empty_effort_passes_no_effort_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_agent_transport: FakeAgentTransport
) -> None:
    fake_agent_transport.queue(fake_agent_transport.success("done"))
    config = '[agent.claude]\neffort = "low"\n[agent.claude.opus]\neffort = ""\n'

    result = _exec(tmp_path, monkeypatch, _AGENT_ASK, home_config=config)

    assert result.exit_code == 0, result.output
    [(_prompt, argv)] = fake_agent_transport.calls
    assert _flag_values(argv, "--effort") == []


def test_exec_never_overrides_an_explicit_thinking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_agent_transport: FakeAgentTransport
) -> None:
    fake_agent_transport.queue(fake_agent_transport.success("done"))
    source = _AGENT_ASK.replace('AgentClaude("opus", "")', 'AgentClaude("opus", "medium")')

    result = _exec(tmp_path, monkeypatch, source, home_config=_OPUS_HIGH)

    assert result.exit_code == 0, result.output
    [(_prompt, argv)] = fake_agent_transport.calls
    assert _flag_values(argv, "--effort") == ["medium"]


def test_exec_keeps_the_agent_value_itself_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_agent_transport: FakeAgentTransport
) -> None:
    fake_agent_transport.queue(fake_agent_transport.success("done"))
    source = (
        'let agent = AgentClaude("opus", "")\n'
        'let answer: text = agent.ask("hi", sandbox = AgentSandbox::Disabled)\n'
        'print (agent == AgentClaude("opus", ""))\n'
    )

    result = _exec(tmp_path, monkeypatch, source, home_config=_OPUS_HIGH)

    assert result.exit_code == 0, result.output
    assert result.output == "true\n"


@pytest.mark.parametrize(
    "config",
    [
        '[agent]\neffort = "high"\n',
        '[agent.claude]\nefort = "high"\n',
        "[agent.claude.opus]\neffort = 3\n",
        '[agent.claude.opus]\nmodel = "x"\n',
        "[agent.pi]\nprovider = 1\n",
        '[agent.opencode]\neffort = "high"\n',
        '[agent.claude.opus.fast]\neffort = "high"\n',
    ],
)
def test_exec_rejects_a_malformed_agent_section_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str
) -> None:
    result = _exec(tmp_path, monkeypatch, 'print "ran"\n', home_config=config)

    assert result.exit_code == 1
    assert "ran" not in result.stdout


def test_agent_config_tables_do_not_configure_the_std_agent_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _exec(tmp_path, monkeypatch, 'import std/agent\nprint "ok"\n', home_config=_OPUS_HIGH)

    assert result.exit_code == 0, result.output
    assert result.output == "ok\n"


def test_a_registered_command_applies_the_configured_effort(
    tmp_path: Path, home: Path, fake_agent_transport: FakeAgentTransport
) -> None:
    source = tmp_path / "source"
    (source / MODULE_TREE_DIRNAME).mkdir(parents=True)
    (source / "package.toml").write_text('[package]\nname = "tools"\nversion = "1.0.0"\n')
    (source / MODULE_TREE_DIRNAME / "main.agl").write_text(
        '@command("tools ask")\n'
        "program def main() -> unit =\n"
        '  let answer: text = AgentClaude("opus", "").ask("hi", sandbox = AgentSandbox::Disabled)\n'
        "  print answer\n"
    )
    install_directory(source, home=home, env={})
    _write_config(home, _OPUS_HIGH)
    fake_agent_transport.queue(fake_agent_transport.success("done"))

    result = _invoke(["tools", "ask"])

    assert result.exit_code == 0, result.output
    assert result.stdout == "done\n"
    [(_prompt, argv)] = fake_agent_transport.calls
    assert _flag_values(argv, "--effort") == ["high"]


def _run_repl(monkeypatch: pytest.MonkeyPatch) -> list[ReplSession]:
    import agm.agl.repl.plain_console as plain_console_mod

    sessions: list[ReplSession] = []

    def capture_session(session: ReplSession, **_kwargs: object) -> None:
        sessions.append(session)

    monkeypatch.setattr(plain_console_mod, "run_plain_console", capture_session)
    repl_command.run(
        ReplArgs(strict_json=None, quiet=False, no_trace=True, trace_file=None, plain=True)
    )
    return sessions


def test_repl_applies_the_configured_effort(
    home: Path, monkeypatch: pytest.MonkeyPatch, fake_agent_transport: FakeAgentTransport
) -> None:
    _write_config(home, _OPUS_HIGH)
    fake_agent_transport.queue(fake_agent_transport.success("done"))

    [session] = _run_repl(monkeypatch)
    result = session.eval_entry(
        'AgentClaude("opus", "").ask("hi", sandbox = AgentSandbox::Disabled)'
    )

    assert result.ok, result.diagnostics
    [(_prompt, argv)] = fake_agent_transport.calls
    assert _flag_values(argv, "--effort") == ["high"]


def test_repl_rejects_a_malformed_agent_section_at_startup(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(home, '[agent.claude]\nefort = "high"\n')

    with pytest.raises(SystemExit) as exc_info:
        _run_repl(monkeypatch)

    assert exc_info.value.code == 1
