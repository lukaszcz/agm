"""Tests for :mod:`agm.agl.repl.themes`."""

from __future__ import annotations

import pytest

from agm.agl.repl.themes import (
    DARK_THEME,
    LIGHT_THEME,
    THEME_NAMES,
    detect_terminal_theme,
    get_style,
)


class TestDetectTerminalTheme:
    def test_colorfgbg_light_background(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COLORFGBG", "0;15")
        assert detect_terminal_theme() == "light"

    def test_colorfgbg_three_part_light(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COLORFGBG", "15;default;15")
        assert detect_terminal_theme() == "light"

    def test_colorfgbg_dark_background(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COLORFGBG", "15;0")
        assert detect_terminal_theme() == "dark"

    def test_colorfgbg_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("COLORFGBG", raising=False)
        assert detect_terminal_theme() == "dark"

    def test_colorfgbg_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COLORFGBG", "")
        assert detect_terminal_theme() == "dark"

    def test_colorfgbg_non_numeric_background(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COLORFGBG", "0;dark")
        assert detect_terminal_theme() == "dark"


class TestGetStyle:
    def test_dark_returns_dark_theme(self) -> None:
        assert get_style("dark") is DARK_THEME

    def test_light_returns_light_theme(self) -> None:
        assert get_style("light") is LIGHT_THEME

    def test_auto_resolves_to_dark_when_env_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("COLORFGBG", raising=False)
        assert get_style("auto") is DARK_THEME

    def test_auto_resolves_to_light_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("COLORFGBG", "0;15")
        assert get_style("auto") is LIGHT_THEME

    def test_unknown_name_falls_back_to_dark(self) -> None:
        assert get_style("unknown") is DARK_THEME


class TestThemeNames:
    def test_contains_expected_names(self) -> None:
        assert set(THEME_NAMES) == {"dark", "light", "auto"}

    def test_is_tuple(self) -> None:
        assert isinstance(THEME_NAMES, tuple)
