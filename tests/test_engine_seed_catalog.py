"""Characterization tests for catalog-driven host engine seeding."""

from __future__ import annotations

import pytest

from agm.agl.runtime.params import engine_default_settings, raw_option_str
from agm.cli_support.engine_seeds import build_host_engine_seeds
from agm.config.engine_keys import ENGINE_KEYS
from agm.config.general import ExecConfig

_CONFIG_RAW_VALUES: dict[str, object] = {
    "log": True,
    "strict-json": True,
    "max-iters": 7,
    "default-agent": 'AgentCommand("configured")',
    "log-file": "configured.jsonl",
    "timeout": "12s",
}

_CLI_VALUES: dict[str, object] = {
    "log": False,
    "strict-json": False,
    "max-iters": 3,
    "log-file": "cli.jsonl",
    "timeout": "3s",
}


def _config_for(key: str, configured: bool) -> ExecConfig:
    return ExecConfig(
        strict_json=configured if key == "strict-json" else False,
        default_loop_limit=7 if configured and key == "max-iters" else None,
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
    cli_values = {key: _CLI_VALUES[key]} if cli_given and key != "default-agent" else {}
    agent = 'AgentCommand("cli")' if cli_given and key == "default-agent" else None

    seeds = build_host_engine_seeds(
        config=_config_for(key, config_given),
        primary_table=primary_table,
        cli_values=cli_values,
        agent=agent,
    )

    actual_keys = set(seeds.values) | set(seeds.overrides)
    expected_keys = {key} if cli_given or config_given else set()
    # A supplied log-file also implies the readable ``log`` setting.
    if key == "log-file" and (cli_given or config_given):
        expected_keys.add("log")
    assert actual_keys == expected_keys


@pytest.mark.parametrize(
    ("key", "raw_value", "expected_keys"),
    [
        ("max-iters", 0, set()),
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
        agent=None,
    )

    assert key not in seeds.values
    assert set(seeds.values) == expected_keys


@pytest.mark.parametrize("key", ["timeout", "log-file"])
def test_explicit_empty_option_cli_value_remains_a_seed(key: str) -> None:
    """Commands encode their negation flags as a present ``None`` value."""
    seeds = build_host_engine_seeds(
        config=_config_for(key, configured=False),
        primary_table={},
        cli_values={key: None},
        agent=None,
    )

    assert key in seeds.values
    assert set(seeds.values) == {key}


def test_invalid_raw_option_value_is_absent() -> None:
    """The raw Option parser leaves non-positive numeric values unset."""
    assert raw_option_str({"timeout": 0}, {}, "timeout") is None


def test_configured_numeric_timeout_is_seeded_from_its_raw_spelling() -> None:
    """Option settings preserve a valid numeric TOML value as text."""
    seeds = build_host_engine_seeds(
        config=ExecConfig(
            strict_json=False,
            default_loop_limit=None,
            timeout=0.5,
            log=False,
            log_file=None,
        ),
        primary_table={"timeout": 0.5},
        cli_values={},
        agent=None,
    )

    assert "timeout" in seeds.values


def test_engine_defaults_cover_exactly_the_catalog_keys_with_defaults() -> None:
    """A catalog addition cannot leave interpreter defaults without a value."""
    expected = {spec.name for spec in ENGINE_KEYS if spec.has_default}
    assert set(engine_default_settings()) == expected
