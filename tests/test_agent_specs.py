"""Pure host-side command construction for typed AgL ``Agent`` values."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agent.spec import AgentClaude, AgentCodex, AgentCommand, AgentPi, AgentSpec
from agm.agl.runtime.agents import decode_agent_value
from agm.agl.semantics.values import IntValue, RecordValue
from tests._agl_helpers import agent_value


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
    assert decode_agent_value(value) == expected


def test_decode_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError):
        decode_agent_value(agent_value("Other", command="runner"))


def test_decode_rejects_non_text_payload_field() -> None:
    value = agent_value("AgentCommand", command="runner")
    value.fields["command"] = IntValue(1)

    with pytest.raises(ValueError):
        decode_agent_value(value)


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
            prompt_via_stdin=True,
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
    from agm.core.process import ProcessCaptureResult

    captured: dict[str, object] = {}

    def fake_run_capture_result(cmd: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured["cmd"] = cmd
        captured["stdin_text"] = kwargs.get("stdin_text")
        return ProcessCaptureResult(
            returncode=0,
            stdout="ok",
            stderr="",
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
            spawn_errno=None,
        )

    monkeypatch.setattr("agm.agent.runner.run_capture_result", fake_run_capture_result)

    temp_files: list[Path] = []
    try:
        prepared = prepare_rendered_prompt_run(
            "rendered prompt text",
            runner=AgentCodex("o3", "high").argv(),
            temp_files=temp_files,
            env={},
            prompt_via_stdin=True,
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
    from agm.core.process import ProcessCaptureResult

    captured: dict[str, object] = {}

    def fake_run_capture_result(cmd: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured["stdin_text"] = kwargs.get("stdin_text")
        return ProcessCaptureResult(
            returncode=0,
            stdout="ok",
            stderr="",
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
            spawn_errno=None,
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
            prompt_via_stdin=True,
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
    from agm.core.process import ProcessCaptureResult

    captured: dict[str, object] = {}

    def fake_run_capture_result(cmd: list[str], **kwargs: object) -> ProcessCaptureResult:
        captured["cmd"] = cmd
        captured["stdin_text"] = kwargs.get("stdin_text")
        return ProcessCaptureResult(
            returncode=0,
            stdout="ok",
            stderr="",
            elapsed=0.1,
            timed_out=False,
            spawn_error=None,
            spawn_errno=None,
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
    from agm.core.process import ProcessCaptureResult

    capture = ProcessCaptureResult(
        returncode=None,
        stdout="out",
        stderr="err",
        elapsed=1.0,
        timed_out=timed_out,
        spawn_error=spawn_error,
        spawn_errno=None,
    )
    monkeypatch.setattr("agm.agent.runner.run_capture_result", lambda *args, **kwargs: capture)
    prepared = PreparedPromptRun(
        command=["runner"], effective_file=Path("prompt.md"), env={}, temp_files=[]
    )

    result = run_prepared_prompt_result(prepared, idle_timeout=None)

    assert result.spawn_error == spawn_error
    assert result.timed_out is timed_out
