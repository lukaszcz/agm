"""AGM home directory and stdlib root environment overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.config.general import agm_home_dir, agm_path_candidates
from agm.config.module_roots import ModuleRootsConfig, resolve_lib_root, resolve_stdlib_root


class TestAgmHomeDir:
    def test_defaults_to_dot_agm_under_home(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        assert agm_home_dir(home=home, env={}) == home / ".agm"

    def test_env_override_replaces_home_dot_agm(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        override = tmp_path / "custom-agm"
        result = agm_home_dir(home=home, env={"AGM_HOME": str(override)})
        assert result == override

    def test_env_override_expands_user(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        result = agm_home_dir(home=home, env={"AGM_HOME": "~/elsewhere"})
        assert result == Path.home() / "elsewhere"

    def test_relative_override_is_made_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.chdir(tmp_path)
        result = agm_home_dir(home=home, env={"AGM_HOME": "relative-agm"})
        assert result.is_absolute()
        assert result == Path.cwd() / "relative-agm"

    def test_blank_override_falls_back_to_default(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        assert agm_home_dir(home=home, env={"AGM_HOME": "   "}) == home / ".agm"

    def test_reads_process_env_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        override = tmp_path / "env-agm"
        monkeypatch.setenv("AGM_HOME", str(override))
        assert agm_home_dir(home=home) == override


class TestAgmPathCandidatesHonorHomeOverride:
    def test_home_candidate_uses_agm_home_override(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        override = tmp_path / "custom-agm"
        candidates = agm_path_candidates(
            home=home, relative_path=Path("config.toml"), env={"AGM_HOME": str(override)}
        )
        assert override / "config.toml" in candidates
        assert home / ".agm" / "config.toml" not in candidates


class TestResolveLibRootEnvOverride:
    def test_agm_home_override_relocates_default_lib_root(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        override = tmp_path / "custom-agm"
        cfg = ModuleRootsConfig(lib_root=None, extra=())

        result = resolve_lib_root(cfg, home=home, env={"AGM_HOME": str(override)})

        assert result == override / "lib"


class TestResolveStdlibRootEnvOverride:
    def test_agm_stdlib_override_wins(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".agm" / "stdlib").mkdir(parents=True)
        override = tmp_path / "my-stdlib"
        override.mkdir()
        result = resolve_stdlib_root(home=home, env={"AGM_STDLIB": str(override)})
        assert result == override

    def test_agm_stdlib_override_expands_user(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        result = resolve_stdlib_root(home=home, env={"AGM_STDLIB": "~/std"})
        assert result == Path.home() / "std"

    def test_relative_override_is_made_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        monkeypatch.chdir(tmp_path)
        result = resolve_stdlib_root(home=home, env={"AGM_STDLIB": "rel-stdlib"})
        assert result.is_absolute()
        assert result == Path.cwd() / "rel-stdlib"

    def test_blank_override_ignored(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        stdlib = home / ".agm" / "stdlib"
        stdlib.mkdir(parents=True)
        assert resolve_stdlib_root(home=home, env={"AGM_STDLIB": ""}) == stdlib

    def test_agm_home_override_relocates_stdlib_candidate(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        override = tmp_path / "custom-agm"
        stdlib = override / "stdlib"
        stdlib.mkdir(parents=True)
        result = resolve_stdlib_root(home=home, env={"AGM_HOME": str(override)})
        assert result == stdlib
