"""Pure host-side command construction for typed AgL ``Agent`` values."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agent.spec import (
    AgentClaude,
    AgentCodex,
    AgentCommand,
    AgentPi,
    AgentSpec,
    build_claude,
    build_codex,
    build_command,
    build_pi,
    decode,
)
from agm.agl.ir.ids import NominalId
from agm.agl.modules.ids import STD_CORE_ID
from agm.agl.semantics.values import EnumValue, IntValue, TextValue


def _value(variant: str, **fields: str) -> EnumValue:
    return EnumValue(
        nominal=NominalId(STD_CORE_ID, "Agent"),
        display_name="Agent",
        variant=variant,
        fields={name: TextValue(field) for name, field in fields.items()},
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            _value("AgentCommand", command="agent --prompt %{PROMPT_FILE}"),
            AgentCommand("agent --prompt %{PROMPT_FILE}"),
        ),
        (_value("AgentClaude", model="sonnet", thinking="high"), AgentClaude("sonnet", "high")),
        (_value("AgentCodex", model="o3", thinking="high"), AgentCodex("o3", "high")),
        (
            _value("AgentPi", provider="openai", model="gpt", thinking="high"),
            AgentPi("openai", "gpt", "high"),
        ),
    ],
)
def test_decode_round_trips_runtime_agent_enum(value: EnumValue, expected: AgentSpec) -> None:
    assert decode(value) == expected


def test_decode_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError):
        decode(_value("Other", command="runner"))


def test_decode_rejects_non_text_payload_field() -> None:
    value = _value("AgentCommand", command="runner")
    value.fields["command"] = IntValue(1)

    with pytest.raises(ValueError):
        decode(value)


def test_kind_builders_include_all_configured_flags_verbatim() -> None:
    assert build_claude(AgentClaude("sonnet", "tool-defined")) == [
        "claude",
        "-p",
        "--model",
        "sonnet",
        "--effort",
        "tool-defined",
    ]
    assert build_codex(AgentCodex("o3", "tool-defined")) == [
        "codex",
        "exec",
        "--model",
        "o3",
        "-c",
        "model_reasoning_effort=tool-defined",
    ]
    assert build_pi(AgentPi("openai", "gpt", "tool-defined")) == [
        "pi",
        "-p",
        "--provider",
        "openai",
        "--model",
        "gpt",
        "--thinking",
        "tool-defined",
    ]


def test_kind_builders_omit_empty_field_flags() -> None:
    assert build_claude(AgentClaude("", "")) == ["claude", "-p"]
    assert build_codex(AgentCodex("", "")) == ["codex", "exec"]
    assert build_pi(AgentPi("", "", "")) == ["pi", "-p"]


def test_built_argv_feeds_shared_prompt_preparation() -> None:
    from agm.agent.runner import cleanup_temp_files, prepare_rendered_prompt_run

    temp_files: list[Path] = []
    try:
        prepared = prepare_rendered_prompt_run(
            "prompt",
            runner=build_claude(AgentClaude("sonnet", "high")),
            temp_files=temp_files,
            env={},
        )
    finally:
        cleanup_temp_files(temp_files)

    assert prepared.command == ["claude", "-p", "--model", "sonnet", "--effort", "high"]


def test_command_builder_preserves_prompt_file_substitution() -> None:
    from agm.agent.runner import command_with_prompt_target

    command = build_command(AgentCommand("runner --input=%{PROMPT_FILE} --copy=%%"))

    assert command_with_prompt_target(command, Path("prompt.md")) == [
        "runner",
        "--input=prompt.md",
        "--copy=prompt.md",
    ]


def test_command_builder_appends_prompt_file_without_placeholder() -> None:
    from agm.agent.runner import command_with_prompt_target

    command = build_command(AgentCommand("runner --quiet"))

    assert command_with_prompt_target(command, Path("prompt.md")) == [
        "runner",
        "--quiet",
        "@prompt.md",
    ]


def test_command_builder_rejects_an_empty_command() -> None:
    with pytest.raises(ValueError):
        build_command(AgentCommand(""))


def test_parse_command_rejects_malformed_shell_words() -> None:
    from agm.agent.runner import parse_command

    with pytest.raises(ValueError):
        parse_command('runner "unterminated', kind="agent")


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
