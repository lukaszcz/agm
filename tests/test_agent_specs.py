"""Pure host-side command construction for typed AgL ``Agent`` values."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agent.runner import PromptDelivery
from agm.agent.spec import AgentClaude, AgentCodex, AgentCommand, AgentPi, AgentSpec, PermissionMode
from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
from agm.agl.runtime.agents import decode_agent_value
from agm.agl.semantics.values import IntValue, RecordValue
from agm.config.general import RunConfig
from agm.sandbox.prepare import SandboxContext, SandboxRun
from agm.sandbox.request import PreparedSandboxCommand, SandboxSpec
from tests._agl_helpers import agent_value


def _sandbox_context(
    tmp_path: Path, *, memory: str | None = None, swap: str | None = None
) -> SandboxContext:
    """A `SandboxContext` with a default settings candidate so `prepare()` succeeds."""
    home = tmp_path / "home"
    cwd = tmp_path / "work"
    (home / ".agm" / "sandbox").mkdir(parents=True, exist_ok=True)
    cwd.mkdir(parents=True, exist_ok=True)
    (home / ".agm" / "sandbox" / "default.json").write_text("{}", encoding="utf-8")
    run_config = RunConfig(
        aliases={},
        default_memory_limit=memory,
        command_memory_limits={},
        default_swap_limit=swap,
        command_swap_limits={},
        default_pty=True,
        command_ptys={},
    )
    return SandboxContext(home=home, proj_dir=None, cwd=cwd, run_config=run_config)


def _sandbox_run(tmp_path: Path, *, profile_name: str = "claude") -> SandboxRun:
    """A `SandboxRun` pairing *profile_name* with a fresh `_sandbox_context`."""
    return SandboxRun(
        spec=SandboxSpec(profile_name=profile_name), context=_sandbox_context(tmp_path)
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            agent_value("AgentCommand", command="agent --prompt %{PROMPT_FILE}"),
            AgentCommand("agent --prompt %{PROMPT_FILE}"),
        ),
        (
            agent_value("AgentClaude", model="sonnet", thinking="high"),
            AgentClaude("sonnet", "high"),
        ),
        (agent_value("AgentCodex", model="o3", thinking="high"), AgentCodex("o3", "high")),
        (
            agent_value("AgentPi", provider="openai", model="gpt", thinking="high"),
            AgentPi("openai", "gpt", "high"),
        ),
    ],
)
def test_decode_round_trips_runtime_agent_enum(value: RecordValue, expected: AgentSpec) -> None:
    assert decode_agent_value(value, NO_BUILTIN_DECLARATIONS) == expected


def test_decode_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError):
        decode_agent_value(agent_value("Other", command="runner"), NO_BUILTIN_DECLARATIONS)


def test_decode_rejects_non_text_payload_field() -> None:
    value = agent_value("AgentCommand", command="runner")
    value.fields["command"] = IntValue(1)

    with pytest.raises(ValueError):
        decode_agent_value(value, NO_BUILTIN_DECLARATIONS)


def test_agent_argv_includes_all_configured_flags_verbatim() -> None:
    assert AgentClaude("sonnet", "tool-defined").argv() == [
        "claude",
        "-p",
        "--model",
        "sonnet",
        "--effort",
        "tool-defined",
    ]
    assert AgentCodex("o3", "tool-defined").argv() == [
        "codex",
        "exec",
        "--model",
        "o3",
        "-c",
        "model_reasoning_effort=tool-defined",
        "-",
    ]
    assert AgentPi("openai", "gpt", "tool-defined").argv() == [
        "pi",
        "-p",
        "--provider",
        "openai",
        "--model",
        "gpt",
        "--thinking",
        "tool-defined",
    ]


def test_agent_argv_omits_empty_field_flags() -> None:
    assert AgentClaude("", "").argv() == ["claude", "-p"]
    assert AgentCodex("", "").argv() == ["codex", "exec", "-"]
    assert AgentPi("", "", "").argv() == ["pi", "-p"]


def test_agent_codex_argv_ends_with_stdin_marker() -> None:
    assert AgentCodex("o3", "high").argv()[-1] == "-"


def test_agent_prompt_via_stdin_flag_per_spec() -> None:
    assert AgentCommand("runner").prompt_via_stdin is False
    assert AgentClaude("sonnet", "high").prompt_via_stdin is False
    assert AgentPi("openai", "gpt", "high").prompt_via_stdin is False
    assert AgentCodex("o3", "high").prompt_via_stdin is True


def test_built_argv_feeds_shared_prompt_preparation() -> None:
    from agm.agent.runner import cleanup_temp_files, prepare_rendered_prompt_run

    temp_files: list[Path] = []
    try:
        prepared = prepare_rendered_prompt_run(
            "prompt",
            runner=AgentClaude("sonnet", "high").argv(),
            temp_files=temp_files,
            env={},
        )
    finally:
        cleanup_temp_files(temp_files)

    assert prepared.command == ["claude", "-p", "--model", "sonnet", "--effort", "high"]
    assert prepared.prompt_via_stdin is False


def test_prepare_rendered_prompt_run_records_the_stdin_delivery_flag() -> None:
    from agm.agent.runner import cleanup_temp_files, prepare_rendered_prompt_run

    temp_files: list[Path] = []
    try:
        prepared = prepare_rendered_prompt_run(
            "prompt",
            runner=AgentCodex("o3", "high").argv(),
            temp_files=temp_files,
            env={},
            delivery=PromptDelivery.STDIN,
        )
    finally:
        cleanup_temp_files(temp_files)

    assert prepared.prompt_via_stdin is True


def test_stdin_delivered_run_sends_prompt_as_stdin_and_appends_no_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prepared codex run sends the rendered prompt via stdin, not ``@<path>``."""
    from agm.agent.runner import (
        cleanup_temp_files,
        prepare_rendered_prompt_run,
        run_prepared_prompt_result,
    )
    from agm.core.process import CapturedOutput, ProcessCaptureResult

    captured: dict[str, object] = {}

    def fake_run_capture_result(cmd: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured["cmd"] = cmd
        captured["stdin_text"] = kwargs.get("stdin_text")
        return ProcessCaptureResult(
            returncode=0,
            stdout=CapturedOutput(data=b"ok", truncated=False),
            stderr=CapturedOutput(data=b"", truncated=False),
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
        )

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)

    temp_files: list[Path] = []
    try:
        prepared = prepare_rendered_prompt_run(
            "rendered prompt text",
            runner=AgentCodex("o3", "high").argv(),
            temp_files=temp_files,
            env={},
            delivery=PromptDelivery.STDIN,
        )
        run_prepared_prompt_result(prepared, idle_timeout=None)
    finally:
        cleanup_temp_files(temp_files)

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert not any(str(element).startswith("@") for element in cmd)
    assert cmd[-1] == "-"
    assert captured["stdin_text"] == "rendered prompt text"


def test_stdin_delivered_prompt_does_not_read_the_prompt_back_off_disk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: the stdin-delivered prompt reaches the child process from
    the already-rendered text handed to ``prepare_rendered_prompt_run``, not by
    writing it to a temp file and reading it back. The old implementation
    silently depended on that temp file surviving between
    ``prepare_rendered_prompt_run`` and ``run_prepared_prompt_result``; no temp
    file is created for this delivery mode at all, and nothing may read one
    back off disk.
    """
    from agm.agent.runner import (
        cleanup_temp_files,
        prepare_rendered_prompt_run,
        run_prepared_prompt_result,
    )
    from agm.core.process import CapturedOutput, ProcessCaptureResult

    captured: dict[str, object] = {}

    def fake_run_capture_result(cmd: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured["stdin_text"] = kwargs.get("stdin_text")
        return ProcessCaptureResult(
            returncode=0,
            stdout=CapturedOutput(data=b"ok", truncated=False),
            stderr=CapturedOutput(data=b"", truncated=False),
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
        )

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)

    def _forbidden_read_text(self: Path, *args: object, **kwargs: object) -> str:
        raise AssertionError("stdin-delivered prompt must not be read back off disk")

    monkeypatch.setattr(Path, "read_text", _forbidden_read_text)

    temp_files: list[Path] = []
    try:
        prepared = prepare_rendered_prompt_run(
            "rendered prompt text",
            runner=AgentCodex("o3", "high").argv(),
            temp_files=temp_files,
            env={},
            delivery=PromptDelivery.STDIN,
        )
        # Nothing needs to read this delivery mode's prompt back off disk, so
        # no temp file should be created for it in the first place.
        assert temp_files == []

        run_prepared_prompt_result(prepared, idle_timeout=None)
    finally:
        cleanup_temp_files(temp_files)

    assert captured["stdin_text"] == "rendered prompt text"


def test_file_delivered_run_is_unchanged_by_the_stdin_delivery_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Claude/Pi (and any non-stdin spec) keep appending ``@<path>`` with no stdin text."""
    from agm.agent.runner import (
        cleanup_temp_files,
        prepare_rendered_prompt_run,
        run_prepared_prompt_result,
    )
    from agm.core.process import CapturedOutput, ProcessCaptureResult

    captured: dict[str, object] = {}

    def fake_run_capture_result(cmd: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured["cmd"] = cmd
        captured["stdin_text"] = kwargs.get("stdin_text")
        return ProcessCaptureResult(
            returncode=0,
            stdout=CapturedOutput(data=b"ok", truncated=False),
            stderr=CapturedOutput(data=b"", truncated=False),
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
        )

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)

    temp_files: list[Path] = []
    try:
        prepared = prepare_rendered_prompt_run(
            "rendered prompt text",
            runner=AgentClaude("sonnet", "high").argv(),
            temp_files=temp_files,
            env={},
        )
        run_prepared_prompt_result(prepared, idle_timeout=None)
    finally:
        cleanup_temp_files(temp_files)

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[-1].startswith("@")
    assert captured["stdin_text"] is None


def test_agent_command_preserves_prompt_file_substitution() -> None:
    from agm.agent.runner import command_with_prompt_target

    command = AgentCommand("runner --input=%{PROMPT_FILE} --copy=%%").argv()

    assert command_with_prompt_target(command, Path("prompt.md"), {}) == [
        "runner",
        "--input=prompt.md",
        "--copy=prompt.md",
    ]


def test_agent_command_appends_prompt_file_without_placeholder() -> None:
    from agm.agent.runner import command_with_prompt_target

    command = AgentCommand("runner --quiet").argv()

    assert command_with_prompt_target(command, Path("prompt.md"), {}) == [
        "runner",
        "--quiet",
        "@prompt.md",
    ]


def test_command_with_prompt_target_can_suppress_the_append_fallback() -> None:
    """A stdin-delivered command still interpolates argv but never gets ``@<target>``."""
    from agm.agent.runner import command_with_prompt_target

    command = AgentCommand("runner --quiet").argv()

    assert (
        command_with_prompt_target(command, Path("prompt.md"), {}, append_target=False) == command
    )


def test_command_with_prompt_target_still_resolves_the_hole_when_append_is_suppressed() -> None:
    """A ``%{PROMPT_FILE}`` hole still resolves even with the ``@<target>`` fallback off."""
    from agm.agent.runner import command_with_prompt_target

    command = AgentCommand("runner --input=%{PROMPT_FILE}").argv()

    assert command_with_prompt_target(command, Path("prompt.md"), {}, append_target=False) == [
        "runner",
        "--input=prompt.md",
    ]


def test_agent_command_rejects_an_empty_command() -> None:
    with pytest.raises(ValueError):
        AgentCommand("").argv()


def test_agent_command_rejects_malformed_shell_words() -> None:
    with pytest.raises(ValueError):
        AgentCommand('runner "unterminated').argv()


@pytest.mark.parametrize("spawn_error,timed_out", [("not found", False), (None, True)])
def test_prepared_runner_maps_process_capture_result(
    monkeypatch: pytest.MonkeyPatch, spawn_error: str | None, timed_out: bool
) -> None:
    from agm.agent.runner import PreparedPromptRun, run_prepared_prompt_result
    from agm.core.process import CapturedOutput, ProcessCaptureResult

    capture = ProcessCaptureResult(
        returncode=None,
        stdout=CapturedOutput(data=b"out", truncated=timed_out),
        stderr=CapturedOutput(data=b"err", truncated=timed_out),
        elapsed=1.0,
        timed_out=timed_out,
        spawn_error=spawn_error,
    )
    monkeypatch.setattr("agm.agent.runner.run_capture_result", lambda *args, **kwargs: capture)
    prepared = PreparedPromptRun(
        command=["runner"], effective_file=Path("prompt.md"), env={}, temp_files=[]
    )

    result = run_prepared_prompt_result(prepared, idle_timeout=None)

    assert result.spawn_error == spawn_error
    assert result.timed_out is timed_out


@pytest.mark.parametrize(
    ("delivery", "argv", "stdin_prompt"),
    [
        (PromptDelivery.FILE, ["runner", "--prepared-file"], None),
        (PromptDelivery.STDIN, ["runner", "--prepared-stdin", "-"], "stdin prompt"),
        (PromptDelivery.LITERAL, ["runner", "--prepared-literal", "literal prompt"], None),
        (PromptDelivery.NONE, ["runner", "--prepared-none"], None),
    ],
)
def test_run_prepared_prompt_honors_prepared_argv_and_stdin_in_normal_mode(
    monkeypatch: pytest.MonkeyPatch,
    delivery: PromptDelivery,
    argv: list[str],
    stdin_prompt: str | None,
) -> None:
    from agm.agent.runner import PreparedPromptRun, run_prepared_prompt

    captured: dict[str, object] = {}

    def fake_run_capture(command: list[str], **kwargs: object) -> tuple[int, str, str]:
        captured["command"] = command
        captured["stdin_text"] = kwargs.get("stdin_text")
        return 0, "output", ""

    monkeypatch.setattr("agm.agent.runner.run_capture", fake_run_capture)
    prepared = PreparedPromptRun(
        command=["runner", "--legacy"],
        effective_file=Path("unused.md"),
        env={},
        temp_files=[],
        stdin_prompt=stdin_prompt,
        argv=argv,
        delivery=delivery,
    )

    assert run_prepared_prompt(prepared) == "output"
    expected_stdin = "" if delivery is PromptDelivery.NONE else stdin_prompt
    assert captured == {"command": argv, "stdin_text": expected_stdin}


def test_prepared_result_closes_stdin_for_no_prompt_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agm.agent.runner import PreparedPromptRun, run_prepared_prompt_result
    from agm.core.process import CapturedOutput, ProcessCaptureResult

    captured: dict[str, object] = {}

    def fake_run_capture_result(argv: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured["argv"] = argv
        captured["stdin_text"] = kwargs.get("stdin_text")
        return ProcessCaptureResult(
            0,
            CapturedOutput(data=b"", truncated=False),
            CapturedOutput(data=b"", truncated=False),
            0.0,
            False,
            None,
        )

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)
    prepared = PreparedPromptRun(
        command=["runner"],
        effective_file=Path("unused.md"),
        env={},
        temp_files=[],
        argv=["runner", "--fork"],
        delivery=PromptDelivery.NONE,
    )

    run_prepared_prompt_result(prepared, idle_timeout=None)

    assert captured == {"argv": ["runner", "--fork"], "stdin_text": ""}


@pytest.mark.parametrize(
    ("delivery", "argv", "stdin_prompt", "formatted_argv"),
    [
        (PromptDelivery.FILE, ["runner", "--prepared-file"], None, "runner --prepared-file"),
        (
            PromptDelivery.STDIN,
            ["runner", "--prepared-stdin", "-"],
            "stdin prompt",
            "runner --prepared-stdin -",
        ),
        (
            PromptDelivery.LITERAL,
            ["runner", "--prepared-literal", "literal prompt"],
            None,
            "runner --prepared-literal 'literal prompt'",
        ),
        (PromptDelivery.NONE, ["runner", "--prepared-none"], None, "runner --prepared-none"),
    ],
)
def test_run_prepared_prompt_honors_prepared_argv_in_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    delivery: PromptDelivery,
    argv: list[str],
    stdin_prompt: str | None,
    formatted_argv: str,
) -> None:
    from agm.agent.runner import PreparedPromptRun, run_prepared_prompt
    from agm.core import dry_run

    monkeypatch.setattr(
        "agm.agent.runner.run_capture",
        lambda *args, **kwargs: pytest.fail("dry run must not execute the command"),
    )
    prepared = PreparedPromptRun(
        command=["runner", "--legacy"],
        effective_file=Path("unused.md"),
        env={},
        temp_files=[],
        stdin_prompt=stdin_prompt,
        argv=argv,
        delivery=delivery,
    )

    dry_run.set_enabled(True)
    try:
        assert run_prepared_prompt(prepared) == ""
    finally:
        dry_run.set_enabled(False)

    assert capsys.readouterr().out == f"dry-run: command [agent]: {formatted_argv}\n"


# ---------------------------------------------------------------------------
# PermissionMode: flags owned by each spec, appended after its own options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "flags"),
    [
        (PermissionMode.NATIVE, ["--permission-mode", "auto"]),
        (PermissionMode.UNRESTRICTED, ["--dangerously-skip-permissions"]),
        (PermissionMode.NONE, []),
    ],
)
def test_claude_argv_appends_permission_flags_after_model_options(
    mode: PermissionMode, flags: list[str]
) -> None:
    assert AgentClaude("sonnet", "high").argv(permission_mode=mode) == [
        "claude",
        "-p",
        "--model",
        "sonnet",
        "--effort",
        "high",
        *flags,
    ]


def test_claude_session_argv_appends_permission_flags_after_model_options() -> None:
    assert AgentClaude("sonnet", "high").session_argv(
        "sid", permission_mode=PermissionMode.UNRESTRICTED
    ) == [
        "claude",
        "-p",
        "--session-id",
        "sid",
        "--model",
        "sonnet",
        "--effort",
        "high",
        "--dangerously-skip-permissions",
    ]


@pytest.mark.parametrize(
    ("mode", "flags"),
    [
        (PermissionMode.NATIVE, ["--approve-for-me"]),
        (PermissionMode.UNRESTRICTED, ["--dangerously-bypass-approvals-and-sandbox"]),
        (PermissionMode.NONE, []),
    ],
)
def test_codex_argv_appends_permission_flags_before_the_stdin_marker(
    mode: PermissionMode, flags: list[str]
) -> None:
    assert AgentCodex("o3", "high").argv(permission_mode=mode) == [
        "codex",
        "exec",
        "--model",
        "o3",
        "-c",
        "model_reasoning_effort=high",
        *flags,
        "-",
    ]


@pytest.mark.parametrize(
    ("mode", "flags"),
    [
        (PermissionMode.NATIVE, ["--approve-for-me"]),
        (PermissionMode.UNRESTRICTED, ["--dangerously-bypass-approvals-and-sandbox"]),
        (PermissionMode.NONE, []),
    ],
)
def test_codex_new_session_argv_appends_permission_flags_before_the_stdin_marker(
    mode: PermissionMode, flags: list[str]
) -> None:
    assert AgentCodex("o3", "high").session_argv(permission_mode=mode) == [
        "codex",
        "exec",
        "--json",
        "--model",
        "o3",
        "-c",
        "model_reasoning_effort=high",
        *flags,
        "-",
    ]


@pytest.mark.parametrize(
    ("mode", "flags"),
    [
        # A resumed thread inherits the approval policy recorded when its
        # session was created, so the native mode re-asserts nothing: the
        # resume form accepts no approval flag. The bypass flag is an explicit
        # escalation and is accepted on both forms.
        (PermissionMode.NATIVE, []),
        (PermissionMode.UNRESTRICTED, ["--dangerously-bypass-approvals-and-sandbox"]),
        (PermissionMode.NONE, []),
    ],
)
def test_codex_resumed_session_argv_re_asserts_no_approval_policy(
    mode: PermissionMode, flags: list[str]
) -> None:
    assert AgentCodex("o3", "high").session_argv("sid", permission_mode=mode) == [
        "codex",
        "exec",
        "resume",
        "sid",
        "--model",
        "o3",
        "-c",
        "model_reasoning_effort=high",
        *flags,
        "-",
    ]


@pytest.mark.parametrize("mode", list(PermissionMode))
def test_pi_adds_no_permission_flags_in_any_mode(mode: PermissionMode) -> None:
    spec = AgentPi("openai", "gpt", "high")
    base = ["pi", "-p", "--provider", "openai", "--model", "gpt", "--thinking", "high"]
    assert spec.argv(permission_mode=mode) == base
    assert spec.session_argv("sid", permission_mode=mode) == [
        "pi",
        "-p",
        "--session-id",
        "sid",
        "--provider",
        "openai",
        "--model",
        "gpt",
        "--thinking",
        "high",
    ]
    assert spec.rpc_argv(permission_mode=mode) == [
        "pi",
        "--mode",
        "rpc",
        "--provider",
        "openai",
        "--model",
        "gpt",
        "--thinking",
        "high",
    ]


@pytest.mark.parametrize("mode", list(PermissionMode))
def test_agent_command_is_verbatim_in_every_permission_mode(mode: PermissionMode) -> None:
    assert AgentCommand("runner --flag").argv(permission_mode=mode) == ["runner", "--flag"]


def test_permission_mode_defaults_to_none_for_unmigrated_callers() -> None:
    """Calling ``argv()`` with no argument keeps today's behavior unchanged."""
    assert AgentClaude("sonnet", "high").argv() == AgentClaude("sonnet", "high").argv(
        permission_mode=PermissionMode.NONE
    )
    assert AgentCodex("o3", "high").argv() == AgentCodex("o3", "high").argv(
        permission_mode=PermissionMode.NONE
    )


# ---------------------------------------------------------------------------
# Runner: sandboxed prepared runs
# ---------------------------------------------------------------------------


def test_agent_call_info_trace_carries_sandbox_and_permission_mode() -> None:
    from agm.agent.transport import AgentCallInfo

    info = AgentCallInfo(
        argv=["claude"],
        prompt_via_stdin=False,
        elapsed=1.0,
        exit_code=0,
        sandboxed=True,
        permission_mode="unrestricted",
    )
    assert info.to_trace() == {
        "argv": ["claude"],
        "prompt_via_stdin": False,
        "elapsed": 1.0,
        "exit_code": 0,
        "sandboxed": True,
        "permission_mode": "unrestricted",
    }


def test_sandboxed_prepared_run_wraps_the_prompt_bearing_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A sandboxed run's argv is ``limits prefix + srt --settings … -- <agent argv>``."""
    from agm.agent.runner import cleanup_temp_files, prepare_rendered_prompt_run

    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")

    temp_files: list[Path] = []
    prepared = prepare_rendered_prompt_run(
        "prompt",
        runner=AgentClaude("sonnet", "high").argv(permission_mode=PermissionMode.UNRESTRICTED),
        temp_files=temp_files,
        env={"PATH": "/bin"},
        sandbox=_sandbox_run(tmp_path, profile_name="claude"),
    )
    try:
        assert isinstance(prepared.sandbox, PreparedSandboxCommand)
        argv = prepared.argv
        assert argv is not None
        assert argv[:4] == ["systemd-run", "--user", "--scope", "-q"]
        srt_index = argv.index("srt")
        assert argv[srt_index : srt_index + 2] == ["srt", "--settings"]
        assert argv[srt_index + 3] == "--"
        tail = argv[srt_index + 4 :]
        assert tail[0:2] == ["claude", "-p"]
        assert "--dangerously-skip-permissions" in tail
        assert tail[-1].startswith("@")
    finally:
        if isinstance(prepared.sandbox, PreparedSandboxCommand):
            prepared.sandbox.close()
        cleanup_temp_files(temp_files)


def test_sandboxed_prepared_run_wraps_a_non_file_delivery_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The non-``FILE`` delivery branch also routes its argv through the sandbox."""
    from agm.agent.runner import cleanup_temp_files, prepare_rendered_prompt_run

    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")

    temp_files: list[Path] = []
    prepared = prepare_rendered_prompt_run(
        "prompt",
        runner=AgentCodex("o3", "high").argv(permission_mode=PermissionMode.UNRESTRICTED),
        temp_files=temp_files,
        env={"PATH": "/bin"},
        delivery=PromptDelivery.STDIN,
        sandbox=_sandbox_run(tmp_path, profile_name="codex"),
    )
    try:
        assert isinstance(prepared.sandbox, PreparedSandboxCommand)
        assert temp_files == []
        argv = prepared.argv
        assert argv is not None
        srt_index = argv.index("srt")
        tail = argv[srt_index + 4 :]
        assert tail[0:2] == ["codex", "exec"]
        assert "--dangerously-bypass-approvals-and-sandbox" in tail
        assert tail[-1] == "-"
        assert not any(part.startswith("@") for part in tail)
    finally:
        if isinstance(prepared.sandbox, PreparedSandboxCommand):
            prepared.sandbox.close()
        cleanup_temp_files(temp_files)


@pytest.mark.parametrize("scenario", ["success", "nonzero_exit", "keyboard_interrupt"])
def test_sandbox_is_closed_on_success_failure_and_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str
) -> None:
    from agm.agent.runner import (
        cleanup_temp_files,
        prepare_rendered_prompt_run,
        run_prepared_prompt_result,
    )
    from agm.core.process import CapturedOutput, ProcessCaptureResult

    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")

    close_calls: list[int] = []
    original_close = PreparedSandboxCommand.close

    def tracking_close(self: PreparedSandboxCommand) -> None:
        close_calls.append(1)
        original_close(self)

    monkeypatch.setattr(PreparedSandboxCommand, "close", tracking_close)

    def fake_run_capture_result(argv: list[str], **kwargs: object) -> ProcessCaptureResult:
        del argv
        if scenario == "keyboard_interrupt":
            raise KeyboardInterrupt
        returncode = 0 if scenario == "success" else 1
        return ProcessCaptureResult(
            returncode=returncode,
            stdout=CapturedOutput(data=b"ok", truncated=False),
            stderr=CapturedOutput(data=b"", truncated=False),
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
        )

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)

    temp_files: list[Path] = []
    try:
        prepared = prepare_rendered_prompt_run(
            "prompt",
            runner=AgentClaude("sonnet", "high").argv(),
            temp_files=temp_files,
            env={"PATH": "/bin"},
            sandbox=_sandbox_run(tmp_path, profile_name="claude"),
        )
        if scenario == "keyboard_interrupt":
            with pytest.raises(KeyboardInterrupt):
                run_prepared_prompt_result(prepared, idle_timeout=None)
        else:
            run_prepared_prompt_result(prepared, idle_timeout=None)
    finally:
        cleanup_temp_files(temp_files)

    assert close_calls == [1]


def test_sandboxed_run_reaches_the_process_primitive_with_the_prepared_wrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The sandboxed argv/env/cwd/cleanup-cmd reaching the process primitive are
    exactly the ones the sandbox library prepared -- not the runner's own
    (unwrapped) argv or ambient env/cwd, and not silently dropped.
    """
    from agm.agent.runner import (
        cleanup_temp_files,
        prepare_rendered_prompt_run,
        run_prepared_prompt_result,
    )
    from agm.core.process import CapturedOutput, ProcessCaptureResult

    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: "/usr/bin/tool")

    captured: dict[str, object] = {}

    def fake_run_capture_result(argv: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured["argv"] = argv
        captured["env"] = kwargs.get("env")
        captured["cwd"] = kwargs.get("cwd")
        captured["interrupt_cleanup_cmd"] = kwargs.get("interrupt_cleanup_cmd")
        return ProcessCaptureResult(
            returncode=0,
            stdout=CapturedOutput(data=b"ok", truncated=False),
            stderr=CapturedOutput(data=b"", truncated=False),
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
        )

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)

    temp_files: list[Path] = []
    try:
        prepared = prepare_rendered_prompt_run(
            "prompt",
            runner=AgentClaude("sonnet", "high").argv(),
            temp_files=temp_files,
            env={"PATH": "/bin"},
            sandbox=_sandbox_run(tmp_path, profile_name="claude"),
        )
        sandbox = prepared.sandbox
        assert isinstance(sandbox, PreparedSandboxCommand)
        run_prepared_prompt_result(prepared, idle_timeout=None)
    finally:
        cleanup_temp_files(temp_files)

    argv = captured["argv"]
    assert argv == prepared.argv
    assert isinstance(argv, list)
    assert argv[:4] == ["systemd-run", "--user", "--scope", "-q"]
    srt_index = argv.index("srt")
    assert argv[srt_index : srt_index + 4] == [
        "srt",
        "--settings",
        str(sandbox.settings_path),
        "--",
    ]

    env = captured["env"]
    assert env is sandbox.env
    assert env is not None
    assert "NODE_USE_ENV_PROXY" in env

    assert captured["cwd"] == sandbox.cwd

    cleanup_cmd = captured["interrupt_cleanup_cmd"]
    assert isinstance(cleanup_cmd, list)
    assert cleanup_cmd[:3] == ["systemctl", "--user", "--no-block"]


def test_sandbox_preparation_failure_becomes_a_spawn_failure_without_stderr_prints(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from agm.agent.runner import (
        SandboxPreparationFailure,
        cleanup_temp_files,
        prepare_rendered_prompt_run,
        prompt_run_result_error,
        run_prepared_prompt_result,
    )

    # `systemd-run` unavailable: the library's resource-limit step fails before
    # any backend or process is touched.
    monkeypatch.setattr("shutil.which", lambda *args, **kwargs: None)

    def fail_run_capture_result(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not spawn a process when sandbox preparation failed")

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fail_run_capture_result)

    temp_files: list[Path] = []
    try:
        prepared = prepare_rendered_prompt_run(
            "prompt",
            runner=AgentClaude("sonnet", "high").argv(),
            temp_files=temp_files,
            env={"PATH": "/bin"},
            sandbox=_sandbox_run(tmp_path, profile_name="claude"),
        )
        assert isinstance(prepared.sandbox, SandboxPreparationFailure)

        result = run_prepared_prompt_result(prepared, idle_timeout=None)
    finally:
        cleanup_temp_files(temp_files)

    assert result.spawn_error is not None
    assert result.returncode is None
    assert result.stdout.text() == ""
    assert result.stderr.text() == ""
    failure = prompt_run_result_error(result)
    assert failure is not None
    assert failure.cause == "spawn_failure"
    assert capsys.readouterr() == ("", "")
