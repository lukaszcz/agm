"""Tests for agm.core.toml."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomlkit
from tomlkit.exceptions import KeyAlreadyPresent, ParseError
from tomlkit.items import Table

from agm.config.general import load_run_config
from agm.core.toml import (
    _get_or_create_table,
    dumps_toml,
    load_toml_doc,
    load_toml_file,
    set_toml_table_value,
    toml_dict,
)

DEPS_SIMPLE = '[deps]\nmylib = "main"\n'
DEPS_OLD = '[deps]\nmylib = "old"\n'
DEPS_EXISTING = '[deps]\nexisting = "branch"\n'
SECTION_DEPS = '[section]\nkey = "value"\n\n[deps]\nlib = "branch"\n'
DEPS_COMMENT = '[deps]  # dependencies\nmylib = "main"\n'
SIMPLE_KEY = 'key = 42\n'


class TestTomlDict:
    def test_dict_is_returned_as_copy(self) -> None:
        d: dict[str, object] = {"key": "value"}
        result = toml_dict(d)
        assert result == {"key": "value"}
        assert result is not d

    def test_none_returns_empty_dict(self) -> None:
        assert toml_dict(None) == {}

    def test_string_returns_empty_dict(self) -> None:
        assert toml_dict("string") == {}

    def test_int_returns_empty_dict(self) -> None:
        assert toml_dict(42) == {}

    def test_list_returns_empty_dict(self) -> None:
        assert toml_dict([1, 2, 3]) == {}

    def test_empty_dict_returns_empty_dict(self) -> None:
        assert toml_dict({}) == {}

    def test_nested_dict_returned_correctly(self) -> None:
        d: dict[str, object] = {"a": {"b": 1}}
        result = toml_dict(d)
        assert result == {"a": {"b": 1}}


class TestLoadTomlFile:
    def test_loads_simple_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(DEPS_SIMPLE, encoding="utf-8")
        result = load_toml_file(toml_file)
        assert result["deps"] == {"mylib": "main"}

    def test_loads_empty_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("", encoding="utf-8")
        result = load_toml_file(toml_file)
        assert result == {}

    def test_loads_nested_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(SECTION_DEPS, encoding="utf-8")
        result = load_toml_file(toml_file)
        assert result["section"] == {"key": "value"}
        assert result["deps"] == {"lib": "branch"}

    def test_returns_plain_dict(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(SIMPLE_KEY, encoding="utf-8")
        result = load_toml_file(toml_file)
        assert isinstance(result, dict)
        assert result["key"] == 42

    def test_raises_on_malformed_toml(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("not = valid = toml = [", encoding="utf-8")
        with pytest.raises(ParseError):
            load_toml_file(toml_file)

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "nonexistent.toml"
        with pytest.raises(FileNotFoundError):
            load_toml_file(toml_file)


class TestLoadTomlDoc:
    def test_returns_toml_document(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(DEPS_SIMPLE, encoding="utf-8")
        doc = load_toml_doc(toml_file)
        assert isinstance(doc, tomlkit.TOMLDocument)
        assert "deps" in doc

    def test_round_trips_preserving_content(self, tmp_path: Path) -> None:
        toml_file = tmp_path / "config.toml"
        original = DEPS_COMMENT
        toml_file.write_text(original, encoding="utf-8")
        doc = load_toml_doc(toml_file)
        output = dumps_toml(doc)
        assert output == original


class TestGetOrCreateTable:
    def test_returns_existing_table(self) -> None:
        doc = tomlkit.parse(DEPS_SIMPLE)
        table = _get_or_create_table(doc, "deps")
        assert isinstance(table, Table)
        assert "mylib" in table

    def test_creates_missing_table(self) -> None:
        doc = tomlkit.document()
        table = _get_or_create_table(doc, "deps")
        assert isinstance(table, Table)
        assert "deps" in doc

    def test_raises_when_existing_value_is_not_a_table(self) -> None:
        """When table_name exists but holds a non-Table value, the isinstance
        check fails and _get_or_create_table attempts doc.add with the same
        key, which tomlkit rejects with KeyAlreadyPresent."""
        doc = tomlkit.document()
        # Insert a plain string under the key that will be queried as a table
        doc.add("deps", "not-a-table")
        with pytest.raises(KeyAlreadyPresent):
            _get_or_create_table(doc, "deps")


class TestSetTomlTableValue:
    def test_creates_table_and_sets_value(self) -> None:
        doc = tomlkit.document()
        set_toml_table_value(doc, "deps", "mylib", "main")
        result = tomlkit.dumps(doc)
        assert "[deps]" in result
        assert "mylib" in result and "main" in result

    def test_updates_existing_key(self) -> None:
        doc = tomlkit.parse(DEPS_OLD)
        set_toml_table_value(doc, "deps", "mylib", "new")
        result = tomlkit.dumps(doc)
        assert "mylib" in result and "new" in result
        assert "old" not in result

    def test_adds_key_to_existing_table(self) -> None:
        doc = tomlkit.parse(DEPS_EXISTING)
        set_toml_table_value(doc, "deps", "newlib", "feat")
        result = tomlkit.dumps(doc)
        assert "existing" in result and "branch" in result
        assert "newlib" in result and "feat" in result


class TestDumpsToml:
    def test_serializes_document(self) -> None:
        doc = tomlkit.document()
        doc.add("section", tomlkit.table())
        doc["section"]["key"] = "value"
        result = dumps_toml(doc)
        assert "[section]" in result
        assert "key" in result and "value" in result


class TestHigherLevelConfigLoaderErrors:
    def test_load_run_config_propagates_malformed_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_run_config propagates a tomlkit ParseError from a malformed config file."""
        home = tmp_path / "home"
        agm_dir = home / ".agm"
        agm_dir.mkdir(parents=True)
        (agm_dir / "config.toml").write_text("not = valid = toml = [", encoding="utf-8")
        monkeypatch.setattr(
            "agm.config.general.agm_installation_prefix", lambda: None
        )
        with pytest.raises(ParseError):
            load_run_config(home=home, proj_dir=None, cwd=tmp_path)
