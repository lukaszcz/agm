"""The ``Agent`` members' ``thinking`` field defaults to ``""`` on every host surface.

AgL construction is covered by ``tests/agl/programs/types/agent_thinking_default.agl``;
these tests cover host-supplied ``Agent`` values omitting ``thinking``: the
``default-agent`` engine setting (CLI flag and config file, decoded against the
seeded type table before any program exists) and an ``Agent``-typed program
parameter (decoded against the program's own declarations).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.runtime.engine_config import convert_config_value
from agm.agl.semantics.type_table import create_seeded_type_table
from agm.agl.semantics.types import BUILTIN_PRELUDE_TYPES
from agm.agl.semantics.values import RecordValue, TextValue
from agm.commands import exec as exec_command
from tests._agl_helpers import write_file_program
from tests.test_exec_command import _config_home, _exec_args_no_trace

_PRINT_AGENTS = (
    "import std/config\n\n"
    "program def main(worker: Agent) -> unit =\n"
    "  print std/config::default-agent\n"
    "  print worker\n"
)
_EXPLICIT = 'AgentCommand("true")'
_CLAUDE = 'Agent::AgentClaude(model = "opus", thinking = "")'
_CODEX = 'Agent::AgentCodex(model = "o3", thinking = "")'
_PI = 'Agent::AgentPi(provider = "anthropic", model = "sonnet", thinking = "")'

_HOST_SPELLINGS = [
    pytest.param('{"$case": "AgentClaude", "model": "opus"}', _CLAUDE, id="json-claude"),
    pytest.param('{"$case": "AgentCodex", "model": "o3"}', _CODEX, id="json-codex"),
    pytest.param(
        '{"$case": "AgentPi", "provider": "anthropic", "model": "sonnet"}', _PI, id="json-pi"
    ),
    pytest.param('AgentClaude(model = "opus")', _CLAUDE, id="named-claude"),
    pytest.param('AgentCodex("o3")', _CODEX, id="positional-codex"),
    pytest.param('AgentPi("anthropic", "sonnet")', _PI, id="positional-pi"),
]


def _run_printing_agents(tmp_path: Path, worker: str, *, default_agent: str | None = None) -> None:
    program = tmp_path / "program.agl"
    write_file_program(program, _PRINT_AGENTS)
    args = _exec_args_no_trace(program, argument_tokens=[worker], default_agent=default_agent)
    exec_command.run(args)


@pytest.mark.parametrize(("spelling", "expected"), _HOST_SPELLINGS)
def test_default_agent_flag_fills_omitted_thinking(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], spelling: str, expected: str
) -> None:
    _run_printing_agents(tmp_path, _EXPLICIT, default_agent=spelling)
    assert capsys.readouterr().out.splitlines()[0] == expected


@pytest.mark.parametrize(("spelling", "expected"), _HOST_SPELLINGS)
def test_agent_program_parameter_fills_omitted_thinking(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], spelling: str, expected: str
) -> None:
    _run_printing_agents(tmp_path, spelling, default_agent=_EXPLICIT)
    assert capsys.readouterr().out.splitlines()[1] == expected


def test_default_agent_config_table_fills_omitted_thinking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config_home(
        tmp_path,
        monkeypatch,
        '[exec]\ndefault-agent = { "$case" = "AgentClaude", model = "opus" }\n',
    )
    _run_printing_agents(tmp_path, _EXPLICIT)
    assert capsys.readouterr().out.splitlines()[0] == _CLAUDE


def test_seeded_agent_decode_fills_omitted_thinking() -> None:
    """The seeded fallback ``Agent`` (no stdlib loaded) defaults ``thinking`` too."""
    value = convert_config_value(
        "default-agent",
        {"$case": "AgentPi", "provider": "anthropic", "model": "sonnet"},
        BUILTIN_PRELUDE_TYPES["Agent"],
        create_seeded_type_table(),
    )
    assert isinstance(value, RecordValue)
    assert value.fields == {
        "provider": TextValue("anthropic"),
        "model": TextValue("sonnet"),
        "thinking": TextValue(""),
    }
