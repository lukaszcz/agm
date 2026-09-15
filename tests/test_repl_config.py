"""Tests for ReplConfig / load_repl_config / save_repl_setting."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from agm.config.general import ReplConfig, load_repl_config, save_repl_setting


class TestReplConfig:
    def test_frozen(self) -> None:
        cfg = ReplConfig(theme="dark")
        with pytest.raises(FrozenInstanceError):
            setattr(cfg, "theme", "light")

    def test_fields(self) -> None:
        cfg = ReplConfig(theme="light", echo=False, echo_unit=True)
        assert cfg.theme == "light"
        assert cfg.echo is False
        assert cfg.echo_unit is True

    def test_default_echo_settings(self) -> None:
        cfg = ReplConfig(theme="auto")
        assert cfg.echo is True
        assert cfg.echo_unit is False


class TestLoadReplConfig:
    def test_defaults_to_auto_when_no_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        cfg = load_repl_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.theme == "auto"

    def test_reads_theme_from_home_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        agm_dir = home / ".agm"
        agm_dir.mkdir(parents=True)
        (agm_dir / "config.toml").write_text('[repl]\ntheme = "light"\n')
        cfg = load_repl_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.theme == "light"

    def test_reads_dark_theme(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        agm_dir = home / ".agm"
        agm_dir.mkdir(parents=True)
        (agm_dir / "config.toml").write_text('[repl]\ntheme = "dark"\n')
        cfg = load_repl_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.theme == "dark"

    def test_invalid_theme_falls_back_to_auto(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        agm_dir = home / ".agm"
        agm_dir.mkdir(parents=True)
        (agm_dir / "config.toml").write_text('[repl]\ntheme = "neon"\n')
        cfg = load_repl_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.theme == "auto"

    def test_project_config_overrides_home(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[repl]\ntheme = "dark"\n')
        proj_dir = tmp_path / "proj"
        (proj_dir / "config").mkdir(parents=True)
        (proj_dir / "config" / "config.toml").write_text('[repl]\ntheme = "light"\n')
        cfg = load_repl_config(home=home, proj_dir=proj_dir, cwd=tmp_path)
        assert cfg.theme == "light"

    def test_echo_settings_default_when_no_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        cfg = load_repl_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.echo is True
        assert cfg.echo_unit is False

    def test_reads_echo_settings_from_home_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        agm_dir = home / ".agm"
        agm_dir.mkdir(parents=True)
        (agm_dir / "config.toml").write_text("[repl]\necho = false\necho-unit = true\n")
        cfg = load_repl_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.echo is False
        assert cfg.echo_unit is True

    def test_invalid_echo_settings_fall_back_to_defaults(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        agm_dir = home / ".agm"
        agm_dir.mkdir(parents=True)
        (agm_dir / "config.toml").write_text('[repl]\necho = "yes"\necho-unit = "no"\n')
        cfg = load_repl_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.echo is True
        assert cfg.echo_unit is False


class TestSaveReplSetting:
    def test_creates_config_file_when_absent(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        save_repl_setting("theme", "dark", home=home)
        config_path = home / ".agm" / "config.toml"
        assert config_path.is_file()
        content = config_path.read_text()
        assert "dark" in content

    def test_written_theme_is_round_trippable(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        save_repl_setting("theme", "light", home=home)
        cfg = load_repl_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.theme == "light"

    def test_written_bool_setting_is_round_trippable(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        save_repl_setting("echo", False, home=home)
        save_repl_setting("echo-unit", True, home=home)
        cfg = load_repl_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.echo is False
        assert cfg.echo_unit is True

    def test_bool_value_round_trips_as_toml_boolean_not_string(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        save_repl_setting("echo", False, home=home)
        content = (home / ".agm" / "config.toml").read_text()
        assert "echo = false" in content
        assert '"false"' not in content

    def test_overwrites_existing_value(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        save_repl_setting("theme", "dark", home=home)
        save_repl_setting("theme", "light", home=home)
        cfg = load_repl_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.theme == "light"

    def test_preserves_other_config_keys(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        agm_dir = home / ".agm"
        agm_dir.mkdir(parents=True)
        (agm_dir / "config.toml").write_text('[exec]\ndefault-agent = "claude -p"\n')
        save_repl_setting("theme", "dark", home=home)
        content = (agm_dir / "config.toml").read_text()
        assert 'default-agent = "claude -p"' in content
        assert "dark" in content

    def test_preserves_other_repl_keys(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        save_repl_setting("theme", "dark", home=home)
        save_repl_setting("echo", False, home=home)
        cfg = load_repl_config(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.theme == "dark"
        assert cfg.echo is False

    def test_creates_parent_dir_when_absent(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        assert not (home / ".agm").exists()
        save_repl_setting("theme", "auto", home=home)
        assert (home / ".agm" / "config.toml").is_file()
