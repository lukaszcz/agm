"""Tests for load_module_roots in src/agm/config/module_roots.py."""

from __future__ import annotations

from pathlib import Path

import pytest

import agm.config.module_roots as module_roots
from agm.config.module_roots import (
    ModuleRootsConfig,
    load_module_roots,
    resolve_lib_root,
    resolve_stdlib_root,
)


class TestModuleRootsConfigConstruction:
    def test_no_lib_root_no_extras(self) -> None:
        cfg = ModuleRootsConfig(lib_root=None, extra=())
        assert cfg.lib_root is None
        assert cfg.extra == ()

    def test_with_lib_root(self, tmp_path: Path) -> None:
        cfg = ModuleRootsConfig(lib_root=("~/.agm/lib", tmp_path), extra=())
        assert cfg.lib_root == ("~/.agm/lib", tmp_path)

    def test_frozen(self, tmp_path: Path) -> None:
        cfg = ModuleRootsConfig(lib_root=None, extra=())
        with pytest.raises((AttributeError, TypeError)):
            setattr(cfg, "lib_root", ("foo", tmp_path))


class TestLoadModuleRootsDefaults:
    def test_no_config_files_returns_none_lib_root_and_empty_extra(
        self, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        cfg = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.lib_root is None
        assert cfg.extra == ()

    def test_default_lib_root_absent_when_no_config(self, tmp_path: Path) -> None:
        """The default ~/.agm/lib is applied by the assembler caller, not load_module_roots."""
        home = tmp_path / "home"
        home.mkdir()
        cfg = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.lib_root is None


class TestLoadModuleRootsFromHomeConfig:
    def test_lib_root_from_home_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        config_file = home / ".agm" / "config.toml"
        config_file.write_text("[modules]\nlib_root = \"/usr/local/agm/lib\"\n")
        cfg = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.lib_root is not None
        raw, origin = cfg.lib_root
        assert raw == "/usr/local/agm/lib"
        # origin is the directory of the config file
        assert origin == config_file.parent

    def test_extra_roots_from_home_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        config_file = home / ".agm" / "config.toml"
        config_file.write_text(
            "[modules]\nroots = [\"/extra/lib1\", \"/extra/lib2\"]\n"
        )
        cfg = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        assert len(cfg.extra) == 2
        raw0, origin0 = cfg.extra[0]
        raw1, origin1 = cfg.extra[1]
        assert raw0 == "/extra/lib1"
        assert raw1 == "/extra/lib2"
        assert origin0 == config_file.parent
        assert origin1 == config_file.parent

    def test_relative_lib_root_origin_is_config_dir(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        config_file = home / ".agm" / "config.toml"
        config_file.write_text("[modules]\nlib_root = \"mylib\"\n")
        cfg = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.lib_root is not None
        raw, origin = cfg.lib_root
        assert raw == "mylib"
        # Caller must resolve relative paths against this origin
        assert origin == home / ".agm"

    def test_relative_extra_root_origin_is_config_dir(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        config_file = home / ".agm" / "config.toml"
        config_file.write_text("[modules]\nroots = [\"./local_lib\"]\n")
        cfg = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        assert len(cfg.extra) == 1
        raw, origin = cfg.extra[0]
        assert raw == "./local_lib"
        assert origin == home / ".agm"


class TestLoadModuleRootsFromProjectConfig:
    def test_lib_root_from_project_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text("[modules]\nlib_root = \"/proj/lib\"\n")
        cfg = load_module_roots(home=home, proj_dir=proj_dir, cwd=tmp_path)
        assert cfg.lib_root is not None
        raw, origin = cfg.lib_root
        assert raw == "/proj/lib"
        assert origin == config_dir

    def test_project_config_overrides_home_lib_root(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            "[modules]\nlib_root = \"/home/lib\"\n"
        )
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text(
            "[modules]\nlib_root = \"/proj/lib\"\n"
        )
        cfg = load_module_roots(home=home, proj_dir=proj_dir, cwd=tmp_path)
        # Project overrides home (same key, last-write-wins via merge)
        assert cfg.lib_root is not None
        raw, _ = cfg.lib_root
        assert raw == "/proj/lib"

    def test_project_extra_roots_use_project_config_dir_as_origin(
        self, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text("[modules]\nroots = [\"../extra\"]\n")
        cfg = load_module_roots(home=home, proj_dir=proj_dir, cwd=tmp_path)
        assert len(cfg.extra) == 1
        raw, origin = cfg.extra[0]
        assert raw == "../extra"
        assert origin == config_dir


class TestLoadModuleRootsFromCwdConfig:
    def test_lib_root_from_cwd_config(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        agm_dir = tmp_path / ".agm"
        agm_dir.mkdir()
        config_file = agm_dir / "config.toml"
        config_file.write_text("[modules]\nlib_root = \"/cwd/lib\"\n")
        cfg = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.lib_root is not None
        raw, origin = cfg.lib_root
        assert raw == "/cwd/lib"
        assert origin == agm_dir

    def test_cwd_config_extra_roots_use_agm_dir_as_origin(
        self, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        agm_dir = tmp_path / ".agm"
        agm_dir.mkdir()
        config_file = agm_dir / "config.toml"
        config_file.write_text("[modules]\nroots = [\"local_lib\"]\n")
        cfg = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        assert len(cfg.extra) == 1
        raw, origin = cfg.extra[0]
        assert raw == "local_lib"
        assert origin == agm_dir


class TestLoadModuleRootsLayering:
    def test_extra_roots_accumulated_across_layers(self, tmp_path: Path) -> None:
        """Extra roots from home + project configs are both included."""
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            "[modules]\nroots = [\"/home/lib\"]\n"
        )
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text(
            "[modules]\nroots = [\"/proj/lib\"]\n"
        )
        cfg = load_module_roots(home=home, proj_dir=proj_dir, cwd=tmp_path)
        raws = [r for r, _ in cfg.extra]
        assert "/home/lib" in raws
        assert "/proj/lib" in raws

    def test_no_modules_section_gives_no_roots(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text("[exec]\nrunner = \"claude\"\n")
        cfg = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.lib_root is None
        assert cfg.extra == ()

    def test_empty_roots_list_gives_no_extras(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text("[modules]\nroots = []\n")
        cfg = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.extra == ()

    def test_whitespace_only_roots_entry_is_skipped(self, tmp_path: Path) -> None:
        """A roots entry that is whitespace-only is ignored, same as an empty string."""
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text(
            "[modules]\nroots = [\"  \", \"/real/path\"]\n"
        )
        cfg = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        # Only the non-whitespace entry survives
        assert len(cfg.extra) == 1
        raw, _ = cfg.extra[0]
        assert raw == "/real/path"


class TestResolveLibRoot:
    """Tests for resolve_lib_root — ensures expanduser and path resolution are correct."""

    def test_tilde_prefixed_lib_root_is_expanded(self, tmp_path: Path) -> None:
        """Regression: a ~-prefixed lib_root must use expanduser, not treated as relative."""
        import os

        origin = tmp_path / "config"
        cfg = ModuleRootsConfig(lib_root=("~/mylib", origin), extra=())
        result = resolve_lib_root(cfg)
        # ~/mylib must expand to an absolute path rooted at the real home dir.
        expected = Path(os.path.expanduser("~/mylib"))
        assert result == expected
        assert result.is_absolute()
        # Must NOT be treated as a path relative to the origin directory.
        assert result != origin / "~/mylib"

    def test_absolute_lib_root_returned_as_is(self, tmp_path: Path) -> None:
        abs_path = tmp_path / "absolute" / "lib"
        cfg = ModuleRootsConfig(lib_root=(str(abs_path), tmp_path / "config"), extra=())
        result = resolve_lib_root(cfg)
        assert result == abs_path

    def test_relative_lib_root_resolved_against_origin(self, tmp_path: Path) -> None:
        origin = tmp_path / "config"
        cfg = ModuleRootsConfig(lib_root=("mylib", origin), extra=())
        result = resolve_lib_root(cfg)
        assert result == origin / "mylib"

    def test_none_lib_root_returns_default_agm_lib(self) -> None:
        """When no lib_root is configured, the default ~/.agm/lib path is returned."""
        import os

        cfg = ModuleRootsConfig(lib_root=None, extra=())
        result = resolve_lib_root(cfg)
        assert result == Path(os.path.expanduser("~/.agm/lib"))
        assert result.is_absolute()


class TestResolveStdlibRoot:
    def test_home_stdlib_is_selected_when_present(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        stdlib = home / ".agm" / "stdlib"
        stdlib.mkdir(parents=True)

        assert resolve_stdlib_root(home=home, env={}) == stdlib

    def test_legacy_home_stdlib_uses_source_tree_fallback(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        stdlib = home / ".agm" / "stdlib" / "std"
        stdlib.mkdir(parents=True)
        (stdlib / "core.agl").write_text("ParsePolicy.Abort", encoding="utf-8")

        result = resolve_stdlib_root(home=home)

        assert result.name == "stdlib"
        assert result.is_dir()
        assert result != stdlib.parent

    def test_missing_home_stdlib_returns_source_tree_fallback(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()

        result = resolve_stdlib_root(home=home, env={})

        assert result.name == "stdlib"
        assert result.is_dir()

    def test_missing_all_stdlib_roots_returns_home_destination(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_module = tmp_path / "pkg" / "src" / "agm" / "config" / "module_roots.py"
        fake_module.parent.mkdir(parents=True)
        fake_module.write_text("")
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(module_roots, "__file__", str(fake_module))

        result = resolve_stdlib_root(home=home, env={})

        assert result == home / ".agm" / "stdlib"
