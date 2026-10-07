"""Interactive agent calls through the ordinary standard-library FFI."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agent.spec_defaults import configured_defaults_resolver
from agm.agl import PipelineDriver
from agm.config.general import load_run_config
from agm.sandbox.prepare import SandboxContext
from tests._agl_helpers import agl_roots, write_sandbox_home, write_transparent_sandbox_shims


@pytest.mark.parametrize(
    ("agent", "expected"),
    [
        ('AgentClaude("sonnet", "high")', ["claude", "--model", "sonnet", "--effort", "high"]),
        (
            'AgentCodex("model", "high")',
            ["codex", "--model", "model", "-c", "model_reasoning_effort=high"],
        ),
        (
            'AgentPi("provider", "model", "high")',
            ["pi", "--provider", "provider", "--model", "model", "--thinking", "high"],
        ),
        ('AgentCommand("worker --prompt \\%{PROMPT_FILE}")', ["worker", "--prompt"]),
    ],
)
@pytest.mark.parametrize("method", [False, True])
def test_chat_hands_the_prompt_and_environment_to_a_foreground_agent(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    agent: str,
    expected: list[str],
    method: bool,
) -> None:
    import agm.core.process as process

    calls: list[tuple[list[str], dict[str, str] | None]] = []
    prompt_files: list[Path] = []

    def foreground(argv: list[str], *, env: dict[str, str] | None = None, **_: object) -> int:
        calls.append((argv, env))
        if argv[0] == "worker":
            file = Path(argv[-1])
            prompt_files.append(file)
            assert file.read_text() == "Help with $HOME and %{literal}"
        else:
            assert argv[-1] == "Help with $HOME and %{literal}"
        print("agent TUI")
        return 0

    monkeypatch.setattr(process, "run_foreground_ignoring_signals", foreground)
    call = (
        "worker.chat(prompt, sandbox = Disabled, env = environment)"
        if method
        else "chat(prompt, worker, Disabled, environment)"
    )
    source = (
        "import std/env::Environ\n"
        "program def main() -> unit =\n"
        f"  let worker = {agent}\n"
        '  let prompt = "Help with $HOME and \\%{literal}"\n'
        '  let environment = Environ({"PATH": "/fake", "MARKER": "chosen"})\n'
        f"  {call}\n"
        '  print "resumed"\n'
    )
    result = PipelineDriver(resolve_agent_spec=None, get_sandbox_context=None).run(
        source, roots=agl_roots()
    )

    assert result.ok, result
    assert capsys.readouterr().out == "agent TUI\nresumed\n"
    assert len(calls) == 1
    argv, environment = calls[0]
    assert argv[:-1] == expected + ([] if argv[0] == "worker" else ["--"])
    assert environment == {"PATH": "/fake", "MARKER": "chosen"}
    assert all(not file.exists() for file in prompt_files)


@pytest.mark.parametrize("outcome", ["nonzero", "missing", "invalid", "interpolation"])
def test_chat_raises_a_catchable_agent_call_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: str,
) -> None:
    import agm.core.process as process

    def foreground(argv: list[str], **_: object) -> int:
        if outcome == "missing":
            raise FileNotFoundError(argv[0])
        return 7

    monkeypatch.setattr(process, "run_foreground_ignoring_signals", foreground)
    command = (
        ""
        if outcome == "invalid"
        else "worker \\%{MISSING}"
        if outcome == "interpolation"
        else "worker"
    )
    result = PipelineDriver(resolve_agent_spec=None, get_sandbox_context=None).run(
        "import std/env::Environ\n"
        "program def main() -> unit =\n"
        "  try\n"
        f'    chat("hello", AgentCommand("{command}"), Disabled, Environ({{}}))\n'
        "  catch AgentCallError as error =>\n"
        "    print error.cause\n",
        roots=agl_roots(),
    )

    assert result.ok, result
    expected = {
        "nonzero": "nonzero_exit",
        "missing": "spawn_failure",
        "invalid": "invalid_agent",
        "interpolation": "interpolation_failure",
    }[outcome]
    assert capsys.readouterr().out == f"{expected}\n"


def test_chat_uses_current_defaults_without_opening_a_persistent_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import agm.core.process as process

    commands: list[list[str]] = []

    def foreground(argv: list[str], **_: object) -> int:
        commands.append(argv)
        return 0

    monkeypatch.setattr(process, "run_foreground_ignoring_signals", foreground)
    resolver = configured_defaults_resolver(
        {"agent": {"claude": {"model": "sonnet", "effort": "high"}}}
    )
    result = PipelineDriver(resolve_agent_spec=resolver, get_sandbox_context=None).run(
        "import std/config\n"
        "program def main() -> unit =\n"
        "  config::default-sandbox := Native\n"
        "  chat()\n"
        "  config::default-agent := AgentCodex()\n"
        "  config::default-sandbox := Disabled\n"
        '  chat("second")\n'
        '  print "resumed"\n',
        roots=agl_roots(),
    )

    assert result.ok, result
    assert commands == [
        ["claude", "--model", "sonnet", "--effort", "high", "--permission-mode", "auto", "--", ""],
        ["codex", "--", "second"],
    ]
    assert capsys.readouterr().out == "resumed\n"


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize(
    ("agent", "permission_flag"),
    [
        ("Claude", "--dangerously-skip-permissions"),
        ("Codex", "--dangerously-bypass-approvals-and-sandbox"),
    ],
)
def test_sandboxed_chat_prepares_a_controlling_terminal_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed: bool,
    agent: str,
    permission_flag: str,
) -> None:
    import agm.core.process as process

    home = tmp_path / "home"
    command = agent.lower()
    write_sandbox_home(home, run_toml=f"[run.{command}]\nmemory = '1G'\nswap = '0'\n")
    binary_dir = tmp_path / "bin"
    write_transparent_sandbox_shims(binary_dir, log_dir=tmp_path / "sandbox-log")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".sandbox").mkdir()
    (tmp_path / ".sandbox" / f"{command}.json").write_text("{}")
    settings: list[Path] = []

    def foreground(argv: list[str], *, env: dict[str, str] | None = None, **options: object) -> int:
        assert "agm.sandbox.pty" in argv
        assert command in argv
        assert permission_flag in argv
        assert ("--no-daemon" in argv) is (agent == "Codex")
        assert "MemoryMax=1G" in argv
        assert env is not None and env["MARKER"] == "chosen"
        assert options["cwd"] == tmp_path
        assert options["interrupt_cleanup_cmd"] is not None
        settings.append(Path(argv[argv.index("--settings") + 1]))
        assert settings[-1].exists()
        if failed:
            raise PermissionError("agent cannot execute")
        return 0

    monkeypatch.setattr(process, "run_foreground_ignoring_signals", foreground)
    context = SandboxContext(
        home=home,
        proj_dir=None,
        cwd=tmp_path,
        run_config=load_run_config(home=home, proj_dir=None, cwd=tmp_path),
    )
    result = PipelineDriver(resolve_agent_spec=None, get_sandbox_context=lambda: context).run(
        "import std/env::Environ\n"
        "program def main() -> unit =\n"
        f'  let environment = Environ({{"PATH": "{binary_dir}", "MARKER": "chosen"}})\n'
        f'  chat("hello", Agent{agent}(), Sandbox, environment)\n',
        roots=agl_roots(),
    )

    if failed:
        assert not result.ok and result.error is not None
        assert result.error.type_name == "AgentCallError"
        assert result.error.fields["cause"] == "spawn_failure"
    else:
        assert result.ok, result
    assert settings and all(not path.exists() for path in settings)
