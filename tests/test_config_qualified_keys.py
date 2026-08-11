"""Tests for qualified AgL configuration key lookup."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from agm.config.general import GeneralConfig, load_general_config
from agm.config.qualified_keys import (
    QualifiedConfigKey,
    QualifiedConfigLookupError,
    resolve_qualified_values,
)
from agm.core.toml import toml_dict


def _config(*layers: Mapping[str, object]) -> GeneralConfig:
    return GeneralConfig.from_layers(toml_dict(layer) for layer in layers)


class TestQualifiedConfigKeys:
    @pytest.mark.parametrize(
        ("table", "expected"),
        [
            ({"judge": {"review": {"max-tries": 1}}}, 1),
            ({"tools": {"judge": {"review": {"max-tries": 2}}}}, 2),
            (
                {"review-tools": {"tools": {"judge": {"review": {"max-tries": 3}}}}},
                3,
            ),
        ],
    )
    def test_resolves_unambiguous_module_suffix_at_each_depth(
        self, table: dict[str, object], expected: int
    ) -> None:
        key = QualifiedConfigKey(("review-tools", "tools", "judge"), ("review",), "max-tries")

        assert resolve_qualified_values(_config(table), (key,)) == {key: expected}

    def test_rejects_table_that_matches_multiple_identities(self) -> None:
        first = QualifiedConfigKey(("review-tools", "judge"), ("review",), "max-tries")
        second = QualifiedConfigKey(("quality-tools", "judge"), ("review",), "max-tries")

        with pytest.raises(QualifiedConfigLookupError) as exc_info:
            resolve_qualified_values(
                _config({"judge": {"review": {"max-tries": 1}}}), (first, second)
            )

        assert "review-tools/judge::review::max-tries" in str(exc_info.value)
        assert "quality-tools/judge::review::max-tries" in str(exc_info.value)

    def test_resolves_distinct_leaves_from_an_otherwise_ambiguous_table(self) -> None:
        first = QualifiedConfigKey(("review-tools", "judge"), ("review",), "max-tries")
        second = QualifiedConfigKey(("quality-tools", "judge"), ("review",), "region")

        assert resolve_qualified_values(
            _config({"judge": {"review": {"max-tries": 1, "region": "eu"}}}),
            (first, second),
        ) == {first: 1, second: "eu"}

    def test_quoted_module_route_is_an_exact_anchor(self) -> None:
        first = QualifiedConfigKey(("review-tools", "judge"), ("review",), "max-tries")
        second = QualifiedConfigKey(("quality-tools", "judge"), ("review",), "max-tries")
        config = {"review-tools/judge": {"review": {"max-tries": 4}}}

        assert resolve_qualified_values(_config(config), (first, second)) == {first: 4}

    def test_rejects_conflicting_spellings_in_one_layer(self) -> None:
        key = QualifiedConfigKey(("review-tools", "judge"), ("review",), "max-tries")
        config = {
            "judge": {"review": {"max-tries": 1}},
            "review-tools": {"judge": {"review": {"max-tries": 2}}},
        }

        with pytest.raises(QualifiedConfigLookupError) as exc_info:
            resolve_qualified_values(_config(config), (key,))

        assert "judge.review" in str(exc_info.value)
        assert "review-tools.judge.review" in str(exc_info.value)

    def test_higher_layer_overrides_a_lower_layer_even_with_a_different_spelling(self) -> None:
        key = QualifiedConfigKey(("review-tools", "judge"), ("review",), "max-tries")
        lower = {"judge": {"review": {"max-tries": 1}}}
        higher = {"review-tools": {"judge": {"review": {"max-tries": 2}}}}

        assert resolve_qualified_values(_config(lower, higher), (key,)) == {key: 2}

    @pytest.mark.parametrize(
        "higher",
        ({"judge": "off"}, {"judge": {"review": "off"}}),
    )
    def test_higher_scalar_removes_a_lower_qualified_value(
        self, higher: Mapping[str, object]
    ) -> None:
        key = QualifiedConfigKey(("review-tools", "judge"), ("review",), "max-tries")
        lower = {"judge": {"review": {"max-tries": 1}}}

        assert resolve_qualified_values(_config(lower, higher), (key,)) == {}

    def test_higher_table_omitting_leaf_retains_lower_qualified_value(self) -> None:
        key = QualifiedConfigKey(("review-tools", "judge"), ("review",), "max-tries")
        lower = {"judge": {"review": {"max-tries": 1}}}
        higher = {"judge": {"review": {"region": "eu"}}}

        assert resolve_qualified_values(_config(lower, higher), (key,)) == {key: 1}

    def test_names_a_quoted_anchor_in_a_same_layer_conflict(self) -> None:
        key = QualifiedConfigKey(("review-tools", "judge"), ("review",), "max-tries")
        config = {
            "judge": {"review": {"max-tries": 1}},
            "review-tools/judge": {"review": {"max-tries": 2}},
        }

        with pytest.raises(QualifiedConfigLookupError) as exc_info:
            resolve_qualified_values(_config(config), (key,))

        assert '["review-tools/judge"].review' in str(exc_info.value)

    @pytest.mark.parametrize(
        "reserved_name",
        ("exec", "wsp", "wt", "deps", "modules", "packages", "params"),
    )
    def test_reserved_sections_are_not_config_module_prefixes(self, reserved_name: str) -> None:
        key = QualifiedConfigKey((reserved_name,), ("review",), "max-tries")
        config = {reserved_name: {"review": {"max-tries": 1}}}

        assert resolve_qualified_values(_config(config), (key,)) == {}

    def test_legacy_params_subtable_cannot_route_to_a_params_module(self) -> None:
        key = QualifiedConfigKey(("params", "workflow"), (), "region")
        config = {"params": {"workflow": {"region": "legacy"}}}

        assert resolve_qualified_values(_config(config), (key,)) == {}

    def test_returns_no_value_when_no_table_matches(self) -> None:
        key = QualifiedConfigKey(("review-tools", "judge"), ("review",), "max-tries")

        assert (
            resolve_qualified_values(_config({"other": {"review": {"max-tries": 1}}}), (key,)) == {}
        )

    def test_ignores_malformed_non_table_paths(self) -> None:
        key = QualifiedConfigKey(("review-tools", "judge"), ("review",), "max-tries")

        assert resolve_qualified_values(_config({"judge": "not a table"}), (key,)) == {}

    def test_one_table_supplies_multiple_leaves_for_one_module_identity(self) -> None:
        first = QualifiedConfigKey(("review-tools", "judge"), ("review",), "region")
        second = QualifiedConfigKey(("review-tools", "judge"), ("review",), "max-tries")
        config = {"judge": {"review": {"region": "eu", "max-tries": 3}}}

        assert resolve_qualified_values(_config(config), (first, second)) == {
            first: "eu",
            second: 3,
        }

    def test_deduplicates_repeated_key_identities(self) -> None:
        key = QualifiedConfigKey(("review-tools", "judge"), ("review",), "max-tries")
        config = {"judge": {"review": {"max-tries": 1}}}

        assert resolve_qualified_values(_config(config), (key, key)) == {key: 1}

    def test_resolves_path_normalized_file_layers_with_dotted_and_quoted_headers(
        self, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        cwd = tmp_path / "workspace"
        home_config = home / ".agm"
        cwd_config = cwd / ".agm"
        home_config.mkdir(parents=True)
        cwd_config.mkdir(parents=True)
        (home_config / "lower.log").touch()
        (cwd_config / "higher.log").touch()
        (home_config / "config.toml").write_text(
            "[judge.review]\nlog-file = 'lower.log'\n", encoding="utf-8"
        )
        (cwd_config / "config.toml").write_text(
            "[\"review-tools/judge\".review]\nlog-file = 'higher.log'\n",
            encoding="utf-8",
        )
        key = QualifiedConfigKey(("review-tools", "judge"), ("review",), "log-file")

        config = load_general_config(home=home, proj_dir=None, cwd=cwd)

        assert config.layers == (
            {"judge": {"review": {"log-file": str(home_config / "lower.log")}}},
            {"review-tools/judge": {"review": {"log-file": str(cwd_config / "higher.log")}}},
        )
        assert resolve_qualified_values(config, (key,)) == {key: str(cwd_config / "higher.log")}
