"""Characterization tests for catalog-driven host engine seeding."""

from __future__ import annotations

import pytest

from agm.agl.ir.builtin_nominals import NO_BUILTIN_DECLARATIONS
from agm.agl.ir.ids import NominalId
from agm.agl.ir.reserved_nominals import (
    require_reserved_enum_member_id,
    require_reserved_nominal_id,
)
from agm.agl.runtime.engine_config import (
    build_engine_config_seeds,
    engine_default_settings,
    raw_option_str,
)
from agm.agl.runtime.option import option_text
from agm.agl.semantics.values import BoolValue, RecordValue, TextValue
from agm.cli_support.engine_seeds import build_host_engine_seeds
from agm.config.engine_keys import (
    ENGINE_KEY_NAMES,
    ENGINE_KEYS,
    ENGINE_REGISTERS,
    TRACE_ENGINE_KEYS,
    TRACE_REGISTER,
    EngineKeySpec,
    trace_write_implies_enabled,
)
from agm.config.general import ExecConfig, exec_config_from_merged
from tests._agl_helpers import agent_value

_CONFIG_RAW_VALUES: dict[str, object] = {
    "trace": True,
    "strict-json": True,
    "default-agent": 'AgentCommand("configured")',
    "default-sandbox": "Native",
    "trace-file": "configured.jsonl",
    "timeout": "12s",
}

_CLI_VALUES: dict[str, object] = {
    "trace": False,
    "strict-json": False,
    "default-agent": 'AgentCommand("cli")',
    "default-sandbox": "Disabled",
    "trace-file": "cli.jsonl",
    "timeout": "3s",
}


def _config_for(key: str, configured: bool) -> ExecConfig:
    return ExecConfig(
        strict_json=configured if key == "strict-json" else False,
        timeout=12.0 if configured and key == "timeout" else None,
        trace=configured if key == "trace" else False,
        trace_file="configured.jsonl" if configured and key == "trace-file" else None,
        default_agent='AgentCommand("configured")'
        if configured and key == "default-agent"
        else None,
        default_sandbox="Native" if configured and key == "default-sandbox" else None,
    )


#: One raw ``[exec]`` TOML value per engine key, paired with the
#: ``ExecConfig`` attribute value ``exec_config_from_merged`` must produce
#: from it. Values are chosen so the hop is meaningful for every key's own
#: parsing (``timeout`` parses a duration string; every other key here is a
#: straight pass-through), never merely a coincidental identity.
_RAW_AND_EXPECTED_BY_KEY: dict[str, tuple[object, object]] = {
    "trace": (True, True),
    "strict-json": (True, True),
    "default-agent": ('AgentCommand("raw-agent")', 'AgentCommand("raw-agent")'),
    "default-sandbox": ("Native", "Native"),
    "trace-file": ("raw.jsonl", "raw.jsonl"),
    "timeout": ("5s", 5.0),
}


@pytest.mark.parametrize("spec", ENGINE_KEYS, ids=lambda spec: spec.name)
def test_exec_config_from_merged_reflects_every_engine_key_config_attr(
    spec: EngineKeySpec,
) -> None:
    """Each ``[exec]`` TOML key reaches its own declared ``ExecConfig`` attribute.

    ``ExecConfig(default_sandbox=...)`` built directly (as :func:`_config_for`
    does) never runs ``exec_config_from_merged``'s own TOML-key lookup, so a
    typo there (e.g. reading the wrong raw key string) would silently leave
    an attribute at its default with nothing failing. Parametrized over the
    whole catalog, so a future engine key is covered automatically.
    """
    assert spec.config_attr is not None
    raw, expected = _RAW_AND_EXPECTED_BY_KEY[spec.name]

    config = exec_config_from_merged({"exec": {spec.name: raw}})

    assert getattr(config, spec.config_attr) == expected


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
    ).merged()

    actual_keys = set(seeds)
    expected_keys = {key} if cli_given or config_given else set()
    # A supplied trace-file also implies the readable ``trace`` setting.
    if key == "trace-file" and (cli_given or config_given):
        expected_keys.add("trace")
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
    ).merged()

    assert seeds["default-agent"] == expected


def test_toml_default_agent_decodes_through_the_same_host_text_dispatch() -> None:
    config = ExecConfig(
        strict_json=False,
        timeout=None,
        trace=False,
        trace_file=None,
        default_agent="claude/sonnet-experimental",
    )

    seeds = build_host_engine_seeds(
        config=config,
        primary_table={"default-agent": config.default_agent},
        cli_values={},
    ).merged()

    assert seeds["default-agent"] == agent_value(
        "AgentClaude", model="sonnet", thinking="experimental"
    )


def test_config_table_default_agent_is_json_shaped_data() -> None:
    """A native TOML table decodes as JSON-shaped data, not host text."""
    raw = {"$case": "AgentClaude", "model": "opus", "thinking": "high"}
    config = ExecConfig(
        strict_json=False, timeout=None, trace=False, trace_file=None, default_agent=raw
    )

    seeds = build_host_engine_seeds(
        config=config, primary_table={"default-agent": raw}, cli_values={}
    ).merged()

    assert seeds["default-agent"] == agent_value("AgentClaude", model="opus", thinking="high")


def _optional_default() -> RecordValue:
    return RecordValue(
        nominal=NominalId(require_reserved_enum_member_id("Optional", "Default")), fields={}
    )


def _option_none() -> RecordValue:
    return RecordValue(
        nominal=NominalId(require_reserved_enum_member_id("Option", "None")), fields={}
    )


def test_bare_default_sandbox_cli_value_fills_every_omitted_field() -> None:
    """A bare ``Sandbox`` fills all four defaulted fields from the host-side constant table.

    No program exists yet at this pre-execution boundary, so the omitted
    fields are filled by :func:`~agm.agl.semantics.type_table.reserved_field_default`
    rather than any evaluator.
    """
    config = _config_for("default-sandbox", configured=False)

    seeds = build_host_engine_seeds(
        config=config, primary_table={}, cli_values={"default-sandbox": "Sandbox"}
    ).merged()

    assert seeds["default-sandbox"] == RecordValue(
        nominal=NominalId(require_reserved_nominal_id("Sandbox")),
        fields={
            "memory": _optional_default(),
            "swap": _optional_default(),
            "settings": _option_none(),
            "patch": BoolValue(True),
        },
    )


def test_partial_default_sandbox_cli_value_keeps_its_explicit_field_and_defaults_the_rest() -> None:
    """An explicit field decodes normally; every omitted field still defaults."""
    config = _config_for("default-sandbox", configured=False)

    seeds = build_host_engine_seeds(
        config=config,
        primary_table={},
        cli_values={"default-sandbox": 'Sandbox(memory = Some("8G"))'},
    ).merged()

    assert seeds["default-sandbox"] == RecordValue(
        nominal=NominalId(require_reserved_nominal_id("Sandbox")),
        fields={
            "memory": RecordValue(
                nominal=NominalId(require_reserved_enum_member_id("Option", "Some")),
                fields={"value": TextValue("8G")},
            ),
            "swap": _optional_default(),
            "settings": _option_none(),
            "patch": BoolValue(True),
        },
    )


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
        strict_json=False, timeout=None, trace=False, trace_file=None, default_agent="   "
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


def test_a_cli_value_shields_a_malformed_configured_value_from_decoding() -> None:
    """A table value the CLI overrides is never decoded, so it cannot fail the run."""
    config = ExecConfig(
        strict_json=False, timeout=None, trace=False, trace_file=None, default_agent="   "
    )

    seeds = build_host_engine_seeds(
        config=config,
        primary_table={"default-agent": "   "},
        fallback_table={"default-agent": "   "},
        cli_values={"default-agent": "worker --flag"},
    ).merged()

    assert seeds["default-agent"] == agent_value("AgentCommand", command="worker --flag")


@pytest.mark.parametrize(
    ("key", "raw_value"),
    [
        ("trace-file", ""),
        ("trace-file", 1),
    ],
)
def test_none_config_values_do_not_suppress_a_builtin_initializer(
    key: str, raw_value: object
) -> None:
    """Invalid/empty config values remain absent instead of becoming seeds.

    Nothing names ``trace`` either, so the derived rule also stays absent.
    """
    seeds = build_host_engine_seeds(
        config=_config_for(key, configured=False),
        primary_table={key: raw_value},
        cli_values={},
    ).merged()

    assert key not in seeds
    assert set(seeds) == set()


@pytest.mark.parametrize("key", ["timeout", "trace-file"])
def test_explicit_empty_option_cli_value_remains_a_seed(key: str) -> None:
    """Commands encode their negation flags as a present ``None`` value.

    A bare ``trace-file`` negation, with no config-table ``trace``/``trace-file``
    to derive from, leaves the derived ``trace`` key absent -- it does not
    manufacture an explicit ``false``.
    """
    seeds = build_host_engine_seeds(
        config=_config_for(key, configured=False),
        primary_table={},
        cli_values={key: None},
    ).merged()

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
            trace=False,
            trace_file=None,
        ),
        primary_table={"timeout": 0.5},
        cli_values={},
    ).merged()

    assert "timeout" in seeds


def test_trace_engine_keys_are_declared_engine_keys() -> None:
    """The trace register pair remains a projection of the engine-key catalog."""
    assert TRACE_ENGINE_KEYS <= ENGINE_KEY_NAMES


def test_every_register_is_named_after_its_own_switch_key() -> None:
    """A register's on/off switch is the key whose name it carries."""
    assert ENGINE_REGISTERS
    for register in ENGINE_REGISTERS:
        switches = [spec for spec in ENGINE_KEYS if spec.name == register]
        assert len(switches) == 1
        assert switches[0].register == register
        assert not switches[0].enables_register


def test_a_some_write_implies_enabled_exactly_for_the_declared_keys() -> None:
    """The runtime implication follows the catalog's ``enables_register`` mark."""
    for spec in ENGINE_KEYS:
        assert trace_write_implies_enabled(spec.name, True) is (
            spec.enables_register and spec.register == TRACE_REGISTER
        )
        assert trace_write_implies_enabled(spec.name, False) is False


def test_path_valued_engine_keys_carry_a_text_option() -> None:
    """A path-valued key is an ``Option[text]``, so a cleared path stays expressible."""
    from agm.config.engine_keys import EngineKeyKind

    paths = [spec for spec in ENGINE_KEYS if spec.is_path]
    assert paths
    assert all(spec.kind is EngineKeyKind.OPTION_TEXT for spec in paths)


def test_engine_defaults_cover_exactly_the_catalog_keys_with_defaults() -> None:
    """A catalog addition cannot leave interpreter defaults without a value."""
    expected = {spec.name for spec in ENGINE_KEYS if spec.has_default}
    assert set(engine_default_settings()) == expected


# ---------------------------------------------------------------------------
# Tier split: cli (explicit CLI flags), upper (primary table), lower
# ([exec]/fallback table) -- independent tiers; cli wins the seeded value.
# ---------------------------------------------------------------------------


def test_a_primary_table_value_is_in_the_upper_tier_and_absent_from_lower() -> None:
    config = ExecConfig(strict_json=True, timeout=None, trace=False, trace_file=None)

    tiers = build_host_engine_seeds(
        config=config, primary_table={"strict-json": True}, cli_values={}
    )

    assert "strict-json" in tiers.upper
    assert "strict-json" not in tiers.lower


def test_a_fallback_table_only_value_is_in_the_lower_tier() -> None:
    config = ExecConfig(strict_json=True, timeout=None, trace=False, trace_file=None)

    tiers = build_host_engine_seeds(
        config=config,
        primary_table={},
        fallback_table={"strict-json": True},
        cli_values={},
    )

    assert "strict-json" in tiers.lower
    assert "strict-json" not in tiers.upper


def test_a_primary_table_value_shadows_the_same_key_in_the_fallback_table() -> None:
    config = ExecConfig(strict_json=True, timeout=None, trace=False, trace_file=None)

    tiers = build_host_engine_seeds(
        config=config,
        primary_table={"strict-json": True},
        fallback_table={"strict-json": False},
        cli_values={},
    )

    assert "strict-json" in tiers.upper
    assert "strict-json" not in tiers.lower


def test_a_cli_value_sits_in_its_own_tier_over_an_undecoded_table_value() -> None:
    """A CLI value sits in ``cli``; the table value it overrides seeds no tier."""
    config = ExecConfig(strict_json=False, timeout=None, trace=False, trace_file=None)

    tiers = build_host_engine_seeds(
        config=config,
        primary_table={},
        fallback_table={"strict-json": True},
        cli_values={"strict-json": False},
    )

    assert "strict-json" in tiers.cli
    assert "strict-json" not in tiers.lower
    assert "strict-json" not in tiers.upper
    assert tiers.merged()["strict-json"] == BoolValue(False)


def test_an_invalid_fallback_only_value_is_absent_from_both_tiers() -> None:
    """A fallback-table leaf that decodes to ``None`` seeds neither tier."""
    config = ExecConfig(strict_json=False, timeout=None, trace=False, trace_file=None)

    tiers = build_host_engine_seeds(
        config=config,
        primary_table={},
        fallback_table={"trace-file": ""},
        cli_values={},
    )

    assert "trace-file" not in tiers.upper
    assert "trace-file" not in tiers.lower


def test_a_named_trace_value_is_seeded_into_its_tier_like_any_other_key() -> None:
    """``trace`` follows the ordinary tier rule; ``merged`` still recomputes it afterward."""
    config = ExecConfig(strict_json=False, timeout=None, trace=True, trace_file=None)

    upper_tiers = build_host_engine_seeds(
        config=config, primary_table={"trace": True}, cli_values={}
    )
    assert "trace" in upper_tiers.upper
    assert "trace" not in upper_tiers.lower

    lower_tiers = build_host_engine_seeds(
        config=config, primary_table={}, fallback_table={"trace": True}, cli_values={}
    )
    assert "trace" in lower_tiers.lower
    assert "trace" not in lower_tiers.upper

    assert upper_tiers.merged()["trace"] == BoolValue(True)


# ---------------------------------------------------------------------------
# ``merged(middle)``: an already-decoded settings tier ranks between ``upper``
# and ``lower``, including for the derived ``trace`` rule.
# ---------------------------------------------------------------------------


def test_merged_places_middle_between_upper_and_lower_for_an_ordinary_key() -> None:
    config = ExecConfig(strict_json=False, timeout=None, trace=False, trace_file=None)
    tiers = build_host_engine_seeds(
        config=config,
        primary_table={},
        fallback_table={"strict-json": False},
        cli_values={},
    )
    middle = build_engine_config_seeds({"strict-json": True})

    assert tiers.merged(middle)["strict-json"] == BoolValue(True)
    assert tiers.merged()["strict-json"] == BoolValue(False)


def test_merged_upper_still_wins_over_middle_for_an_ordinary_key() -> None:
    config = ExecConfig(strict_json=True, timeout=None, trace=False, trace_file=None)
    tiers = build_host_engine_seeds(
        config=config, primary_table={"strict-json": True}, cli_values={}
    )
    middle = build_engine_config_seeds({"strict-json": False})

    assert tiers.merged(middle)["strict-json"] == BoolValue(True)


def test_a_middle_trace_file_with_no_higher_trace_seeds_trace_true() -> None:
    """A middle-tier ``trace-file`` with nothing above it still implies ``trace``."""
    config = ExecConfig(strict_json=False, timeout=None, trace=False, trace_file=None)
    tiers = build_host_engine_seeds(config=config, primary_table={}, cli_values={})
    middle = build_engine_config_seeds({"trace-file": "trace.jsonl"})

    assert tiers.merged(middle)["trace"] == BoolValue(True)
    assert "trace" not in tiers.merged()


def test_program_table_trace_beats_middle_trace_but_middle_trace_file_still_applies() -> None:
    """``trace`` picks the program table's explicit value; ``trace-file`` still wins from middle.

    ``trace`` and ``trace-file`` resolve their highest tier independently: the
    program table (``upper``) outranks ``middle`` for ``trace`` even though
    ``middle`` is the only tier setting ``trace-file``.
    """
    config = ExecConfig(strict_json=False, timeout=None, trace=False, trace_file=None)
    tiers = build_host_engine_seeds(config=config, primary_table={"trace": False}, cli_values={})
    middle = build_engine_config_seeds({"trace": True, "trace-file": "trace.jsonl"})

    assert tiers.merged(middle)["trace"] == BoolValue(True)


def test_program_table_trace_false_and_exec_trace_file_still_seed_trace_true() -> None:
    """Regression: program ``trace = false`` plus ``[exec] trace-file`` still implies ``trace``.

    ``ExecConfig`` already folds the program table over ``[exec]`` for
    ``trace``/``trace_file``; the tier split must not change this outcome.
    """
    config = ExecConfig(strict_json=False, timeout=None, trace=False, trace_file="p")
    tiers = build_host_engine_seeds(
        config=config,
        primary_table={"trace": False},
        fallback_table={"trace-file": "p"},
        cli_values={},
    )

    assert tiers.merged()["trace"] == BoolValue(True)


def test_an_invalid_program_table_trace_file_with_a_middle_trace_file_seeds_trace_true() -> None:
    """Regression: the merged ``trace-file`` value and the derived ``trace`` must agree.

    A program-table ``trace-file`` that fails to decode is absent from the
    value merge, so ``trace-file`` in the result comes entirely from ``middle``
    -- ``trace`` must be derived from that same merged value, not from raw
    table membership.
    """
    config = ExecConfig(strict_json=False, timeout=None, trace=False, trace_file=None)
    tiers = build_host_engine_seeds(config=config, primary_table={"trace-file": ""}, cli_values={})
    middle = build_engine_config_seeds({"trace-file": "trace.jsonl"})

    merged = tiers.merged(middle)
    assert (
        merged["trace-file"]
        == build_engine_config_seeds({"trace-file": "trace.jsonl"})["trace-file"]
    )
    assert merged["trace"] == BoolValue(True)


def test_cli_trace_wins_over_a_middle_tier_trace_file() -> None:
    config = ExecConfig(strict_json=False, timeout=None, trace=False, trace_file=None)
    tiers = build_host_engine_seeds(config=config, primary_table={}, cli_values={"trace": False})
    middle = build_engine_config_seeds({"trace-file": "trace.jsonl"})

    assert tiers.merged(middle)["trace"] == BoolValue(False)


def test_trace_stays_absent_when_nothing_configures_it_anywhere_including_middle() -> None:
    config = ExecConfig(strict_json=False, timeout=None, trace=False, trace_file=None)
    tiers = build_host_engine_seeds(config=config, primary_table={}, cli_values={})
    middle = build_engine_config_seeds({"strict-json": True})

    assert "trace" not in tiers.merged(middle)


def test_a_middle_trace_file_of_none_hides_a_lower_tier_trace_file() -> None:
    """A middle tier explicitly setting ``trace-file`` to ``None`` outranks ``[exec]``."""
    fallback = {"trace-file": "from-exec.jsonl"}
    config = exec_config_from_merged({"exec": fallback})
    tiers = build_host_engine_seeds(
        config=config, primary_table={}, fallback_table=fallback, cli_values={}
    )
    middle = build_engine_config_seeds({"trace-file": None})

    assert tiers.merged(middle)["trace"] == BoolValue(False)


@pytest.mark.parametrize(
    ("primary", "fallback"),
    [
        ({"trace-file": "path.jsonl"}, {}),
        ({}, {"trace-file": "path.jsonl"}),
    ],
    ids=["program_table_trace_file", "exec_trace_file"],
)
def test_cli_no_trace_file_does_not_suppress_a_configured_trace_file(
    primary: dict[str, object], fallback: dict[str, object]
) -> None:
    """``--no-trace-file`` clears only the CLI seed; a configured path still traces.

    Documented in docs/agl/reference/host-environment.md ("--no-trace-file
    semantics") and docs/commands/agl.md: ``--no-trace-file`` clears the
    initial CLI ``trace-file`` value only -- it does not suppress a trace a
    program-table or ``[exec]`` ``trace-file`` configures. The seeded
    ``trace-file`` register still reflects the CLI negation (``None``); the
    derived ``trace`` register still comes on because the config value
    independently enables tracing.
    """
    config = exec_config_from_merged({"exec": fallback}, program_table=primary)
    tiers = build_host_engine_seeds(
        config=config,
        primary_table=primary,
        fallback_table=fallback,
        cli_values={"trace-file": None},
    )

    merged = tiers.merged()
    assert merged["trace"] == BoolValue(True)
    assert option_text(merged["trace-file"], nominals=NO_BUILTIN_DECLARATIONS) is None


@pytest.mark.parametrize(
    ("primary", "fallback", "cli_values"),
    [
        ({"trace": "yes"}, {}, {}),
        ({}, {"trace-file": "   "}, {}),
        ({"trace": False}, {"trace-file": "trace.jsonl"}, {}),
        ({}, {"trace": True}, {"trace-file": None}),
    ],
    ids=[
        "invalid_trace_value_in_program_table",
        "whitespace_only_trace_file_in_exec",
        "program_trace_false_and_exec_trace_file",
        "cli_trace_file_negation_with_configured_trace",
    ],
)
def test_merged_trace_matches_config_trace_or_trace_file_is_set(
    primary: dict[str, object],
    fallback: dict[str, object],
    cli_values: dict[str, object | None],
) -> None:
    """``merged()`` names ``trace`` as ``cfg.trace or cfg.trace_file is not None`` when configured.

    Builds ``cfg`` through ``exec_config_from_merged`` -- the same path a real
    caller uses -- rather than a hand-built ``ExecConfig``, so this pins the
    derived rule against the real config layering. A fully invalid/unnamed
    combination leaves ``trace`` absent instead of an explicit ``false`` --
    the same effective setting, since that is the builtin default.
    """
    config = exec_config_from_merged({"exec": fallback}, program_table=primary)

    tiers = build_host_engine_seeds(
        config=config, primary_table=primary, fallback_table=fallback, cli_values=cli_values
    )

    assert tiers.merged().get("trace", BoolValue(False)) == BoolValue(
        config.trace or config.trace_file is not None
    )


def test_trace_stays_absent_when_neither_trace_nor_trace_file_is_configured_anywhere() -> None:
    config = exec_config_from_merged({"exec": {}}, program_table={})

    tiers = build_host_engine_seeds(
        config=config, primary_table={}, fallback_table={}, cli_values={}
    )

    assert "trace" not in tiers.merged()
