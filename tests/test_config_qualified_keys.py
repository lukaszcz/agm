"""Tests for qualified AgL configuration key lookup."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from agm.config.general import GeneralConfig, load_general_config
from agm.config.qualified_keys import (
    QualifiedConfigKey,
    QualifiedConfigLookupError,
    configured_leaf_table_candidates,
    configured_leaf_tables,
    resolve_manifest_values,
    resolve_qualified_values,
    route_table_paths,
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
    def test_reserved_sections_are_not_unscoped_config_modules(self, reserved_name: str) -> None:
        key = QualifiedConfigKey((reserved_name,), (), "max-tries")
        config = {reserved_name: {"max-tries": 1}}

        assert resolve_qualified_values(_config(config), (key,)) == {}

    @pytest.mark.parametrize("schema_name", ("deps", "modules", "packages", "params"))
    def test_schema_sections_are_not_config_modules_at_any_depth(self, schema_name: str) -> None:
        """A section keyed by AGM's own schema never doubles as a program table."""
        key = QualifiedConfigKey((schema_name,), ("main",), "region")
        config = {schema_name: {"main": {"region": "configured"}}}

        assert resolve_qualified_values(_config(config), (key,)) == {}

    @pytest.mark.parametrize("command_name", ("exec", "wsp", "wt"))
    def test_command_sections_still_serve_a_loose_files_program_table(
        self, command_name: str
    ) -> None:
        key = QualifiedConfigKey((command_name,), ("main",), "region")
        config = {command_name: {"main": {"region": "configured"}}}

        assert resolve_qualified_values(_config(config), (key,)) == {key: "configured"}

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

    def test_route_table_paths_share_a_table_between_a_suffix_and_its_own_route(self) -> None:
        """A nested module route reads the table its last segment names."""
        entry = route_table_paths(("demo",))
        imported = route_table_paths(("lib", "demo"))

        assert entry == (("demo",),)
        assert set(entry).issubset(imported)

    def test_scope_region_route_includes_suffixes_and_the_quoted_module_anchor(self) -> None:
        """A scope-region leaf reads the same qualified table spellings as its binding."""
        assert route_table_paths(("A", "logging"), ("debug",)) == (
            ("logging", "debug"),
            ("A", "logging", "debug"),
            ("A/logging", "debug"),
        )

    def test_resolves_a_registered_command_path_table(self) -> None:
        """A command path addresses the program its package registers under it."""
        key = QualifiedConfigKey(
            ("review-tools", "review"), ("main",), "strict", command_paths=(("dev", "review"),)
        )

        assert resolve_qualified_values(_config({"dev": {"review": {"strict": True}}}), (key,)) == {
            key: True
        }

    def test_rejects_a_command_path_table_conflicting_with_its_module_route(self) -> None:
        key = QualifiedConfigKey(
            ("review-tools", "judge"), ("main",), "strict", command_paths=(("dev", "review"),)
        )
        config = _config(
            {"dev": {"review": {"strict": True}}, "judge": {"main": {"strict": False}}}
        )

        with pytest.raises(QualifiedConfigLookupError) as exc_info:
            resolve_qualified_values(config, (key,))

        assert "conflicting tables" in str(exc_info.value)

    def test_a_later_layer_command_path_table_overrides_a_module_route(self) -> None:
        key = QualifiedConfigKey(
            ("review-tools", "judge"), ("main",), "strict", command_paths=(("dev", "review"),)
        )
        config = _config(
            {"judge": {"main": {"strict": False}}}, {"dev": {"review": {"strict": True}}}
        )

        assert resolve_qualified_values(config, (key,)) == {key: True}

    def test_resolves_every_command_path_registered_for_one_program(self) -> None:
        """Two registrations of one program are two spellings of its address."""
        first = QualifiedConfigKey(
            ("tools", "run"), ("main",), "level", command_paths=(("dev", "run"), ("dev", "r"))
        )

        assert resolve_qualified_values(_config({"dev": {"r": {"level": "high"}}}), (first,)) == {
            first: "high"
        }

    def test_command_path_table_does_not_duplicate_a_module_route_spelling(self) -> None:
        """A command path that coincides with a module route stays one table."""
        paths = route_table_paths(("tools", "judge"), ("main",), (("judge", "main"),))

        assert paths.count(("judge", "main")) == 1
        assert resolve_qualified_values(
            _config({"judge": {"main": {"level": "high"}}}),
            (
                QualifiedConfigKey(
                    ("tools", "judge"), ("main",), "level", command_paths=(("judge", "main"),)
                ),
            ),
        ) == {
            QualifiedConfigKey(
                ("tools", "judge"), ("main",), "level", command_paths=(("judge", "main"),)
            ): "high"
        }

    def test_a_schema_rooted_command_path_has_no_config_table(self) -> None:
        """A command registered under a schema section (``agent review``) reads no
        ``[agent.review]`` or inherited ``[agent]`` table, which belong to AGM's schema."""
        key = QualifiedConfigKey(
            ("tools", "review"), ("main",), "strict", command_paths=(("agent", "review"),)
        )
        config = _config({"agent": {"strict": True, "review": {"strict": True}}})

        assert ("agent", "review") not in route_table_paths(
            key.module_segments, key.scope_path, key.command_paths
        )
        assert resolve_qualified_values(config, (key,)) == {}
        assert (
            configured_leaf_table_candidates(
                config, key.module_segments, key.scope_path, key.command_paths
            )
            == {}
        )

    def test_configured_leaf_tables_report_command_path_leaves(self) -> None:
        config = _config({"dev": {"review": {"strict": True}}})

        assert configured_leaf_tables(
            config, ("review-tools", "review"), ("main",), command_paths=(("dev", "review"),)
        ) == {"strict": ("dev", "review")}

    def test_configured_leaf_tables_report_every_consulted_spelling(self) -> None:
        config = _config(
            {"judge": {"review": {"max-tries": 1}}},
            {"review-tools/judge": {"review": {"region": "eu"}}},
        )

        assert configured_leaf_tables(config, ("review-tools", "judge"), ("review",)) == {
            "max-tries": ("judge", "review"),
            "region": ("review-tools/judge", "review"),
        }

    def test_configured_leaf_tables_exclude_nested_tables(self) -> None:
        config = _config({"workflow": {"msg": "hi", "main": {"strict-json": True}}})

        assert configured_leaf_tables(config, ("workflow",)) == {"msg": ("workflow",)}

    def test_configured_leaf_tables_skip_reserved_sections(self) -> None:
        config = _config({"exec": {"strict-json": True}})

        assert configured_leaf_tables(config, ("exec",)) == {}

    def test_configured_leaf_tables_are_empty_without_a_matching_table(self) -> None:
        config = _config({"other": {"region": "eu"}})

        assert configured_leaf_tables(config, ("workflow",)) == {}

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
            "[judge.review]\ntrace-file = 'lower.log'\n", encoding="utf-8"
        )
        (cwd_config / "config.toml").write_text(
            "[\"review-tools/judge\".review]\ntrace-file = 'higher.log'\n",
            encoding="utf-8",
        )
        key = QualifiedConfigKey(("review-tools", "judge"), ("review",), "trace-file")

        config = load_general_config(home=home, proj_dir=None, cwd=cwd)

        assert config.layers == (
            {"judge": {"review": {"trace-file": str(home_config / "lower.log")}}},
            {"review-tools/judge": {"review": {"trace-file": str(cwd_config / "higher.log")}}},
        )
        assert resolve_qualified_values(config, (key,)) == {key: str(cwd_config / "higher.log")}


class TestGroupTableInheritance:
    """A command path's proper prefixes are inherited config tables."""

    def test_a_group_table_feeds_a_command_beneath_it(self) -> None:
        key = QualifiedConfigKey(
            ("tools", "review"), ("main",), "strict", command_paths=(("devel", "review"),)
        )

        assert resolve_qualified_values(_config({"devel": {"strict": True}}), (key,)) == {key: True}

    def test_a_deeper_group_table_wins_over_a_shallower_one(self) -> None:
        key = QualifiedConfigKey(
            ("tools", "review"),
            ("main",),
            "strict",
            command_paths=(("devel", "sub", "review"),),
        )
        config = _config({"devel": {"strict": False, "sub": {"strict": True}}})

        assert resolve_qualified_values(config, (key,)) == {key: True}

    def test_the_exact_route_beats_every_inherited_group_table(self) -> None:
        key = QualifiedConfigKey(
            ("tools", "review"), ("main",), "strict", command_paths=(("devel", "review"),)
        )
        config = _config(
            {"devel": {"review": {"strict": True}, "strict": False}},
        )

        assert resolve_qualified_values(config, (key,)) == {key: True}

    def test_specificity_beats_layer_order(self) -> None:
        """An earlier, deeper table beats a later, shallower one."""
        key = QualifiedConfigKey(
            ("tools", "review"),
            ("main",),
            "x",
            command_paths=(("devel", "sub", "review"),),
        )
        earlier = {"devel": {"sub": {"x": 1}}}
        later = {"devel": {"x": 2}}

        assert resolve_qualified_values(_config(earlier, later), (key,)) == {key: 1}

    def test_a_later_layer_overrides_an_earlier_one_within_one_inherited_tier(self) -> None:
        key = QualifiedConfigKey(
            ("tools", "review"), ("main",), "strict", command_paths=(("devel", "review"),)
        )

        assert resolve_qualified_values(
            _config({"devel": {"strict": False}}, {"devel": {"strict": True}}), (key,)
        ) == {key: True}

    def test_alias_prefixes_are_already_expanded_command_paths(self) -> None:
        """An alias's own group table (``[dev]``) inherits down just like the canonical one."""
        key = QualifiedConfigKey(
            ("tools", "review"),
            ("main",),
            "strict",
            command_paths=(("dev", "review"), ("devel", "review")),
        )

        assert resolve_qualified_values(_config({"dev": {"strict": True}}), (key,)) == {key: True}

    def test_same_depth_conflict_from_two_command_paths_in_one_layer_errors(self) -> None:
        key = QualifiedConfigKey(
            ("tools", "review"),
            ("main",),
            "strict",
            command_paths=(("devel", "review"), ("staging", "review")),
        )
        config = _config({"devel": {"strict": True}, "staging": {"strict": False}})

        with pytest.raises(QualifiedConfigLookupError):
            resolve_qualified_values(config, (key,))

    def test_a_conflict_in_a_tier_shadowed_by_a_higher_one_still_errors(self) -> None:
        """A higher tier's value winning does not exempt a lower tier from its own conflicts."""
        key = QualifiedConfigKey(
            ("tools", "review"),
            ("main",),
            "x",
            command_paths=(("dev", "review"), ("devel", "review")),
        )
        config = _config({"dev": {"x": 1, "review": {"x": 3}}, "devel": {"x": 2}})

        with pytest.raises(QualifiedConfigLookupError):
            resolve_qualified_values(config, (key,))

    def test_a_later_layer_non_tabular_exact_route_falls_through_to_an_inherited_tier(
        self,
    ) -> None:
        key = QualifiedConfigKey(
            ("tools", "review"), ("main",), "strict", command_paths=(("devel", "review"),)
        )
        config = _config(
            {"devel": {"review": {"strict": True}, "strict": False}},
            {"devel": {"review": "off"}},
        )

        assert resolve_qualified_values(config, (key,)) == {key: False}

    def test_a_later_layer_non_tabular_replacement_drops_an_inherited_value(self) -> None:
        key = QualifiedConfigKey(
            ("tools", "review"), ("main",), "strict", command_paths=(("devel", "review"),)
        )

        assert (
            resolve_qualified_values(_config({"devel": {"strict": True}}, {"devel": "off"}), (key,))
            == {}
        )

    def test_no_command_paths_means_no_inheritance(self) -> None:
        key = QualifiedConfigKey(("tools", "review"), ("main",), "strict")

        assert resolve_qualified_values(_config({"tools": {"strict": True}}), (key,)) == {}

    def test_configured_leaf_tables_never_reports_inherited_tables(self) -> None:
        config = _config({"devel": {"strict": True}})

        assert (
            configured_leaf_tables(
                config,
                ("tools", "review"),
                ("main",),
                command_paths=(("devel", "review"),),
            )
            == {}
        )

    def test_configured_leaf_table_candidates_always_includes_inherited_tables(self) -> None:
        config = _config({"devel": {"strict": True}})

        assert configured_leaf_table_candidates(
            config,
            ("tools", "review"),
            ("main",),
            command_paths=(("devel", "review"),),
        ) == {"strict": frozenset({("devel",)})}

    def test_configured_leaf_table_candidates_collects_every_table_across_layers(self) -> None:
        """Unlike configured_leaf_tables, every table a leaf was ever read from is kept."""
        config = _config({"devel": {"review": {"strict": True}}}, {"devel": {"strict": False}})

        assert configured_leaf_table_candidates(
            config,
            ("tools", "review"),
            ("main",),
            command_paths=(("devel", "review"),),
        ) == {"strict": frozenset({("devel", "review"), ("devel",)})}


class TestResolveManifestValues:
    """A package manifest ``[config]`` table resolves command > groups > root."""

    def test_a_command_table_wins_over_a_group_table(self) -> None:
        key = QualifiedConfigKey(
            ("tools", "review"), ("main",), "strict", command_paths=(("devel", "review"),)
        )
        manifest = {"devel": {"strict": False, "review": {"strict": True}}}

        assert resolve_manifest_values(manifest, (key,)) == {key: True}

    def test_a_group_table_wins_over_the_root(self) -> None:
        key = QualifiedConfigKey(
            ("tools", "review"), ("main",), "strict", command_paths=(("devel", "review"),)
        )
        manifest = {"strict": False, "devel": {"strict": True}}

        assert resolve_manifest_values(manifest, (key,)) == {key: True}

    def test_a_deeper_group_table_wins_over_a_shallower_one(self) -> None:
        key = QualifiedConfigKey(
            ("tools", "review"),
            ("main",),
            "strict",
            command_paths=(("devel", "sub", "review"),),
        )
        manifest = {"devel": {"strict": False, "sub": {"strict": True}}}

        assert resolve_manifest_values(manifest, (key,)) == {key: True}

    def test_a_program_with_no_command_paths_reads_only_the_root(self) -> None:
        key = QualifiedConfigKey(("tools", "review"), ("main",), "strict")
        manifest = {"strict": True, "devel": {"strict": False}}

        assert resolve_manifest_values(manifest, (key,)) == {key: True}

    def test_no_root_and_no_command_table_resolves_nothing(self) -> None:
        key = QualifiedConfigKey(
            ("tools", "review"), ("main",), "strict", command_paths=(("devel", "review"),)
        )

        assert resolve_manifest_values({"devel": {"other": 1}}, (key,)) == {}

    def test_a_leaf_alias_resolves_like_its_primary_spelling(self) -> None:
        key = QualifiedConfigKey(
            ("tools", "logging"),
            ("main",),
            "verbose",
            command_paths=(("devel", "review"),),
            leaf_aliases=("logging.verbose",),
        )
        manifest = {"devel": {"review": {"logging.verbose": True}}}

        assert resolve_manifest_values(manifest, (key,)) == {key: True}

    def test_same_tier_conflicting_spellings_error(self) -> None:
        key = QualifiedConfigKey(
            ("tools", "logging"),
            ("main",),
            "verbose",
            command_paths=(("devel", "review"),),
            leaf_aliases=("logging.verbose",),
        )
        manifest = {"devel": {"review": {"verbose": True, "logging.verbose": False}}}

        with pytest.raises(QualifiedConfigLookupError):
            resolve_manifest_values(manifest, (key,))

    def test_a_module_suffix_route_never_addresses_a_manifest_table(self) -> None:
        """Only command paths and the root apply; a bare module name is not a route."""
        key = QualifiedConfigKey(("tools", "review"), ("main",), "strict")

        assert resolve_manifest_values({"tools": {"strict": True}}, (key,)) == {}

    def test_a_root_tier_conflict_reports_cleanly_instead_of_crashing(self) -> None:
        """A program with no command table still gets a clean error, not a raw ``ValueError``."""
        key = QualifiedConfigKey(
            ("tools", "logging"), ("main",), "verbose", leaf_aliases=("logging.verbose",)
        )
        manifest = {"verbose": True, "logging.verbose": False}

        with pytest.raises(QualifiedConfigLookupError, match=r"config\.verbose"):
            resolve_manifest_values(manifest, (key,))
