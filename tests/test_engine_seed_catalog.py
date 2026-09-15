"""Characterization tests for catalog-driven host engine seeding."""

from __future__ import annotations

import pytest

from agm.agl.runtime.engine_config import engine_default_settings, raw_option_str
from agm.cli_support.engine_seeds import build_host_engine_seeds
from agm.config.engine_keys import ENGINE_KEY_NAMES, ENGINE_KEYS, TRACE_ENGINE_KEYS
from agm.config.general import ExecConfig
from tests._agl_helpers import agent_value

_CONFIG_RAW_VALUES: dict[str, object] = {
    "log": True,
    "strict-json": True,
    "default-agent": 'AgentCommand("configured")',
    "log-file": "configured.jsonl",
    "timeout": "12s",
}

_CLI_VALUES: dict[str, object] = {
    "log": False,
    "strict-json": False,
    "default-agent": 'AgentCommand("cli")',
    "log-file": "cli.jsonl",
    "timeout": "3s",
}


def _config_for(key: str, configured: bool) -> ExecConfig:
    return ExecConfig(
        strict_json=configured if key == "strict-json" else False,
        timeout=12.0 if configured and key == "timeout" else None,
        log=configured if key == "log" else False,
        log_file="configured.jsonl" if configured and key == "log-file" else None,
        default_agent='AgentCommand("configured")'
        if configured and key == "default-agent"
        else None,
    )


@pytest.mark.parametrize("key", [spec.name for spec in ENGINE_KEYS])
@pytest.mark.parametrize(
    ("cli_given", "config_given"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_each_engine_key_seed_has_the_same_cli_config_presence_matrix(
    key: str, cli_given: bool, config_given: bool
) -> None:
    """Every catalog key is seeded exactly when a host layer supplied it."""
    primary_table = {key: _CONFIG_RAW_VALUES[key]} if config_given else {}
    cli_values = {key: _CLI_VALUES[key]} if cli_given else {}

    seeds = build_host_engine_seeds(
        config=_config_for(key, config_given),
        primary_table=primary_table,
        cli_values=cli_values,
    )

    actual_keys = set(seeds)
    expected_keys = {key} if cli_given or config_given else set()
    # A supplied log-file also implies the readable ``log`` setting.
    if key == "log-file" and (cli_given or config_given):
        expected_keys.add("log")
    assert actual_keys == expected_keys


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("claude/sonnet-medium", agent_value("AgentClaude", model="sonnet", thinking="medium")),
        ("codex/o3-high", agent_value("AgentCodex", model="o3", thinking="high")),
        (
            "pi/openai/gpt-5-low",
            agent_value("AgentPi", provider="openai", model="gpt-5", thinking="low"),
        ),
        (
            "anthropic/claude-opus-custom",
            agent_value("AgentPi", provider="anthropic", model="claude-opus", thinking="custom"),
        ),
        ("worker --flag", agent_value("AgentCommand", command="worker --flag")),
        (
            'AgentClaude("opus", "high")',
            agent_value("AgentClaude", model="opus", thinking="high"),
        ),
        (
            'Agent::AgentPi(provider = "openai", model = "gpt-5", thinking = "low")',
            agent_value("AgentPi", provider="openai", model="gpt-5", thinking="low"),
        ),
        (
            '{"$case": "AgentCommand", "command": "echo hi"}',
            agent_value("AgentCommand", command="echo hi"),
        ),
    ],
)
def test_cli_agent_values_decode_to_the_typed_agent_value(raw: str, expected: object) -> None:
    seeds = build_host_engine_seeds(
        config=_config_for("default-agent", configured=False),
        primary_table={},
        cli_values={"default-agent": raw},
    )

    assert seeds["default-agent"] == expected


def test_toml_default_agent_decodes_through_the_same_host_text_dispatch() -> None:
    config = ExecConfig(
        strict_json=False,
        timeout=None,
        log=False,
        log_file=None,
        default_agent="claude/sonnet-experimental",
    )

    seeds = build_host_engine_seeds(
        config=config,
        primary_table={"default-agent": config.default_agent},
        cli_values={},
    )

    assert seeds["default-agent"] == agent_value(
        "AgentClaude", model="sonnet", thinking="experimental"
    )


def test_config_table_default_agent_is_json_shaped_data() -> None:
    """A native TOML table decodes as JSON-shaped data, not host text."""
    raw = {"$case": "AgentClaude", "model": "opus", "thinking": "high"}
    config = ExecConfig(
        strict_json=False, timeout=None, log=False, log_file=None, default_agent=raw
    )

    seeds = build_host_engine_seeds(
        config=config, primary_table={"default-agent": raw}, cli_values={}
    )

    assert seeds["default-agent"] == agent_value("AgentClaude", model="opus", thinking="high")


def test_invalid_default_agent_value_exits_before_anything_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A decode failure prints an error naming ``--default-agent`` and exits 1."""
    with pytest.raises(SystemExit) as exc_info:
        build_host_engine_seeds(
            config=_config_for("default-agent", configured=False),
            primary_table={},
            cli_values={"default-agent": 'AgentClaude(model = "x"'},
        )
    assert exc_info.value.code == 1
    assert "--default-agent" in capsys.readouterr().err


def test_invalid_configured_default_agent_value_exits_naming_the_config_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A malformed program-table value names ``default-agent``, never the flag or ``[exec]``.

    ``primary_table`` here stands for a resolved ``[prog.main]`` table --
    ``exec_config_from_merged`` has already applied program-table-over-
    ``[exec]`` precedence into ``config.default_agent`` by the time
    ``build_host_engine_seeds`` runs.
    """
    config = ExecConfig(
        strict_json=False, timeout=None, log=False, log_file=None, default_agent="   "
    )

    with pytest.raises(SystemExit) as exc_info:
        build_host_engine_seeds(
            config=config, primary_table={"default-agent": "   "}, cli_values={}
        )
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "default-agent" in err
    assert "--default-agent" not in err
    assert "[exec]" not in err


@pytest.mark.parametrize(
    ("key", "raw_value", "expected_keys"),
    [
        ("log-file", "", {"log"}),
        ("log-file", 1, {"log"}),
    ],
)
def test_none_config_values_do_not_suppress_a_builtin_initializer(
    key: str, raw_value: object, expected_keys: set[str]
) -> None:
    """Invalid/empty config values remain absent instead of becoming seeds."""
    seeds = build_host_engine_seeds(
        config=_config_for(key, configured=False),
        primary_table={key: raw_value},
        cli_values={},
    )

    assert key not in seeds
    assert set(seeds) == expected_keys


@pytest.mark.parametrize("key", ["timeout", "log-file"])
def test_explicit_empty_option_cli_value_remains_a_seed(key: str) -> None:
    """Commands encode their negation flags as a present ``None`` value."""
    seeds = build_host_engine_seeds(
        config=_config_for(key, configured=False),
        primary_table={},
        cli_values={key: None},
    )

    assert key in seeds
    assert set(seeds) == {key}


def test_invalid_raw_option_value_is_absent() -> None:
    """The raw Option parser leaves non-positive numeric values unset."""
    assert raw_option_str({"timeout": 0}, {}, "timeout") is None


def test_configured_numeric_timeout_is_seeded_from_its_raw_spelling() -> None:
    """Option settings preserve a valid numeric TOML value as text."""
    seeds = build_host_engine_seeds(
        config=ExecConfig(
            strict_json=False,
            timeout=0.5,
            log=False,
            log_file=None,
        ),
        primary_table={"timeout": 0.5},
        cli_values={},
    )

    assert "timeout" in seeds


def test_trace_engine_keys_are_declared_engine_keys() -> None:
    """The trace register pair remains a projection of the engine-key catalog."""
    assert TRACE_ENGINE_KEYS <= ENGINE_KEY_NAMES


def test_engine_defaults_cover_exactly_the_catalog_keys_with_defaults() -> None:
    """A catalog addition cannot leave interpreter defaults without a value."""
    expected = {spec.name for spec in ENGINE_KEYS if spec.has_default}
    assert set(engine_default_settings()) == expected
