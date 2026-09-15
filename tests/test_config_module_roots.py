"""Tests for load_module_roots in src/agm/config/module_roots.py."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import semver

import agm.stdlib_locator as stdlib_locator
from agm.config.module_roots import (
    ModuleRootsConfig,
    StdlibResolutionError,
    StdlibVersionMismatchError,
    load_module_roots,
    resolve_lib_root,
    resolve_stdlib_root,
)
from agm.packages.activation import ActivationIndex, ActivePackage, write_activation_index
from agm.packages.record import write_record
from agm.version import AGM_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _activate_stdlib(home: Path, version: str) -> Path:
    """Create and activate a store ``std`` package at *version*."""
    root = home / ".agm" / "packages" / "std" / version
    shutil.copytree(_REPO_ROOT / "packages" / "stdlib", root)
    write_record(root)
    write_activation_index(
        ActivationIndex({"std": ActivePackage(semver.Version.parse(version))}),
        home=home,
        env={},
    )
    return root


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
    def test_no_config_files_returns_none_lib_root_and_empty_extra(self, tmp_path: Path) -> None:
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
        config_file.write_text('[modules]\nlib_root = "/usr/local/agm/lib"\n')
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
        config_file.write_text('[modules]\nroots = ["/extra/lib1", "/extra/lib2"]\n')
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
        config_file.write_text('[modules]\nlib_root = "mylib"\n')
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
        config_file.write_text('[modules]\nroots = ["./local_lib"]\n')
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
        config_file.write_text('[modules]\nlib_root = "/proj/lib"\n')
        cfg = load_module_roots(home=home, proj_dir=proj_dir, cwd=tmp_path)
        assert cfg.lib_root is not None
        raw, origin = cfg.lib_root
        assert raw == "/proj/lib"
        assert origin == config_dir

    def test_project_config_overrides_home_lib_root(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text('[modules]\nlib_root = "/home/lib"\n')
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text('[modules]\nlib_root = "/proj/lib"\n')
        cfg = load_module_roots(home=home, proj_dir=proj_dir, cwd=tmp_path)
        # Project overrides home (same key, last-write-wins via merge)
        assert cfg.lib_root is not None
        raw, _ = cfg.lib_root
        assert raw == "/proj/lib"

    def test_project_extra_roots_use_project_config_dir_as_origin(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text('[modules]\nroots = ["../extra"]\n')
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
        config_file.write_text('[modules]\nlib_root = "/cwd/lib"\n')
        cfg = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        assert cfg.lib_root is not None
        raw, origin = cfg.lib_root
        assert raw == "/cwd/lib"
        assert origin == agm_dir

    def test_cwd_config_extra_roots_use_agm_dir_as_origin(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        agm_dir = tmp_path / ".agm"
        agm_dir.mkdir()
        config_file = agm_dir / "config.toml"
        config_file.write_text('[modules]\nroots = ["local_lib"]\n')
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
        (home / ".agm" / "config.toml").write_text('[modules]\nroots = ["/home/lib"]\n')
        proj_dir = tmp_path / "proj"
        config_dir = proj_dir / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text('[modules]\nroots = ["/proj/lib"]\n')
        cfg = load_module_roots(home=home, proj_dir=proj_dir, cwd=tmp_path)
        raws = [r for r, _ in cfg.extra]
        assert "/home/lib" in raws
        assert "/proj/lib" in raws

    def test_no_modules_section_gives_no_roots(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        (home / ".agm").mkdir(parents=True)
        (home / ".agm" / "config.toml").write_text("[exec]\nstrict-json = true\n")
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
        (home / ".agm" / "config.toml").write_text('[modules]\nroots = ["  ", "/real/path"]\n')
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

    def test_interpolates_lib_root_and_extra_roots_leniently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        config_dir = home / ".agm"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text(
            '[modules]\nlib_root = "%{PROJ}/lib"\nroots = ["%{PROJ}/a", "plain/b", "%{MISSING}"]\n'
        )
        monkeypatch.setenv("PROJ", str(tmp_path / "project"))

        config = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)

        assert resolve_lib_root(config) == tmp_path / "project" / "lib"
        assert [raw for raw, _ in config.extra] == [
            str(tmp_path / "project" / "a"),
            "plain/b",
            "%{MISSING}",
        ]

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
    def test_matching_active_store_stdlib_is_selected(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        store_stdlib = _activate_stdlib(home, AGM_VERSION)

        assert resolve_stdlib_root(home=home, env={}) == store_stdlib.resolve()

    def test_incompatible_active_store_stdlib_requires_the_managed_refresh(
        self, tmp_path: Path
    ) -> None:
        """An installed library with the legacy method registry cannot be selected."""
        home = tmp_path / "home"
        installed_version = "0.1.1"
        assert installed_version != AGM_VERSION
        _activate_stdlib(home, installed_version)

        with pytest.raises(StdlibVersionMismatchError):
            resolve_stdlib_root(home=home, env={})

    def test_active_store_stdlib_version_mismatch_names_versions_and_remediation(
        self, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        installed_version = "0.0.1"
        _activate_stdlib(home, installed_version)

        with pytest.raises(StdlibVersionMismatchError) as exc_info:
            resolve_stdlib_root(home=home, env={})

        message = str(exc_info.value)
        assert installed_version in message
        assert AGM_VERSION in message
        assert "just install" in message

    def test_missing_active_store_tree_falls_back_to_repo_checkout(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        write_activation_index(
            ActivationIndex({"std": ActivePackage(semver.Version.parse(AGM_VERSION))}),
            home=home,
            env={},
        )

        assert resolve_stdlib_root(home=home, env={}) == _REPO_ROOT / "packages" / "stdlib"

    def test_unreadable_active_store_stdlib_manifest_reports_a_resolution_error(
        self, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        store_stdlib = _activate_stdlib(home, AGM_VERSION)
        (store_stdlib / "package.toml").write_text("not valid = [", encoding="utf-8")

        with pytest.raises(StdlibResolutionError, match="manifest"):
            resolve_stdlib_root(home=home, env={})

    def test_malformed_activation_is_a_stdlib_resolution_error(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        index_path = home / ".agm" / "packages" / "index.toml"
        index_path.parent.mkdir(parents=True)
        index_path.write_text("not valid = [", encoding="utf-8")

        with pytest.raises(StdlibResolutionError, match="activation"):
            resolve_stdlib_root(home=home, env={})

    def test_editable_active_stdlib_is_rejected(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        editable = tmp_path / "editable"
        editable.mkdir()
        write_activation_index(
            ActivationIndex(
                {"std": ActivePackage(semver.Version.parse(AGM_VERSION), editable=editable)}
            ),
            home=home,
            env={},
        )

        with pytest.raises(StdlibResolutionError, match="editable"):
            resolve_stdlib_root(home=home, env={})

    def test_active_stdlib_store_path_rejects_a_symlink(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        store = home / ".agm" / "packages"
        target = tmp_path / "target"
        target.mkdir()
        store.mkdir(parents=True)
        (store / "std").symlink_to(target, target_is_directory=True)
        write_activation_index(
            ActivationIndex({"std": ActivePackage(semver.Version.parse(AGM_VERSION))}),
            home=home,
            env={},
        )

        with pytest.raises(StdlibResolutionError, match="symlink"):
            resolve_stdlib_root(home=home, env={})

    def test_active_stdlib_store_path_must_be_a_directory(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        store_stdlib = home / ".agm" / "packages" / "std" / AGM_VERSION
        store_stdlib.parent.mkdir(parents=True)
        store_stdlib.touch()
        write_activation_index(
            ActivationIndex({"std": ActivePackage(semver.Version.parse(AGM_VERSION))}),
            home=home,
            env={},
        )

        with pytest.raises(StdlibResolutionError, match="directory"):
            resolve_stdlib_root(home=home, env={})

    def test_active_stdlib_manifest_must_match_the_activation(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        store_stdlib = _activate_stdlib(home, AGM_VERSION)
        (store_stdlib / "package.toml").write_text(
            '[package]\nname = "other"\nversion = "1.0.0"\n', encoding="utf-8"
        )

        with pytest.raises(StdlibResolutionError, match="activation"):
            resolve_stdlib_root(home=home, env={})

    def test_agm_stdlib_override_is_returned_unchecked(self, tmp_path: Path) -> None:
        synthetic_stdlib = tmp_path / "synthetic_stdlib"
        synthetic_stdlib.mkdir()

        result = resolve_stdlib_root(
            home=tmp_path / "home", env={"AGM_STDLIB": str(synthetic_stdlib)}
        )

        assert result == synthetic_stdlib

    def test_missing_active_store_stdlib_falls_back_to_repo_checkout(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()

        assert resolve_stdlib_root(home=home, env={}) == _REPO_ROOT / "packages" / "stdlib"

    def test_missing_shipped_stdlib_fails_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        locator = tmp_path / "site-packages" / "agm" / "stdlib_locator.py"
        locator.parent.mkdir(parents=True)
        monkeypatch.setattr(stdlib_locator, "__file__", str(locator))

        with pytest.raises(StdlibResolutionError, match="shipped"):
            resolve_stdlib_root(home=tmp_path / "home", env={})

    def test_clean_wheel_context_falls_back_to_bundled_stdlib(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        locator = tmp_path / "site-packages" / "agm" / "stdlib_locator.py"
        bundled_stdlib = locator.parent / "stdlib"
        bundled_stdlib.mkdir(parents=True)
        monkeypatch.setattr(stdlib_locator, "__file__", str(locator))

        assert resolve_stdlib_root(home=tmp_path / "home", env={}) == bundled_stdlib


def _write_std_checkout(root: Path, version: str) -> Path:
    """Create a development ``std`` package checkout at *root*."""
    (root / "src").mkdir(parents=True)
    (root / "package.toml").write_text(
        f'[package]\nname = "std"\nversion = "{version}"\n', encoding="utf-8"
    )
    return root


class TestResolveStdlibRootFromAnchor:
    """The ``std`` checkout containing the anchor is the standard library."""

    def test_development_std_checkout_containing_the_anchor_is_selected(
        self, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        checkout = _write_std_checkout(tmp_path / "checkout", AGM_VERSION)

        selected = resolve_stdlib_root(home=home, env={}, anchor=checkout / "src" / "agent.agl")

        assert selected == checkout.resolve()

    def test_development_std_checkout_outranks_a_mismatched_active_store_package(
        self, tmp_path: Path
    ) -> None:
        """Editing a checkout must not be blocked by a stale activated store tree."""
        home = tmp_path / "home"
        _activate_stdlib(home, "0.0.1")
        checkout = _write_std_checkout(tmp_path / "checkout", "9.9.9")

        selected = resolve_stdlib_root(home=home, env={}, anchor=checkout / "src" / "agent.agl")

        assert selected == checkout.resolve()

    def test_anchor_inside_the_immutable_store_keeps_store_selection(self, tmp_path: Path) -> None:
        """A store tree is owned by activation, so it is never a development checkout."""
        home = tmp_path / "home"
        store_stdlib = _activate_stdlib(home, "0.0.1")
        # ``_activate_stdlib`` copies the repository manifest verbatim; make the
        # store tree self-consistent so it is recognized by its canonical path.
        (store_stdlib / "package.toml").write_text(
            '[package]\nname = "std"\nversion = "0.0.1"\n', encoding="utf-8"
        )

        with pytest.raises(StdlibVersionMismatchError):
            resolve_stdlib_root(home=home, env={}, anchor=store_stdlib / "src" / "prelude.agl")

    def test_agm_stdlib_override_outranks_a_development_std_checkout(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        checkout = _write_std_checkout(tmp_path / "checkout", AGM_VERSION)
        override = tmp_path / "override"
        override.mkdir()

        selected = resolve_stdlib_root(
            home=home,
            env={"AGM_STDLIB": str(override)},
            anchor=checkout / "src" / "agent.agl",
        )

        assert selected == override

    def test_anchor_in_a_non_std_development_package_keeps_store_selection(
        self, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        store_stdlib = _activate_stdlib(home, AGM_VERSION)
        alpha = tmp_path / "alpha"
        (alpha / "src").mkdir(parents=True)
        (alpha / "package.toml").write_text(
            '[package]\nname = "alpha"\nversion = "1.0.0"\n', encoding="utf-8"
        )

        selected = resolve_stdlib_root(home=home, env={}, anchor=alpha / "src" / "main.agl")

        assert selected == store_stdlib.resolve()

    def test_unreadable_anchor_manifest_keeps_store_selection(self, tmp_path: Path) -> None:
        """A broken manifest is reported by package discovery, not by stdlib selection."""
        home = tmp_path / "home"
        store_stdlib = _activate_stdlib(home, AGM_VERSION)
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / "package.toml").write_text("not valid = [", encoding="utf-8")

        selected = resolve_stdlib_root(home=home, env={}, anchor=broken / "main.agl")

        assert selected == store_stdlib.resolve()

    def test_std_package_whose_modules_exclude_the_anchor_is_not_the_standard_library(
        self, tmp_path: Path
    ) -> None:
        """An unrelated package named ``std`` stays inert for the files beside it."""
        home = tmp_path / "home"
        store_stdlib = _activate_stdlib(home, AGM_VERSION)
        unrelated = _write_std_checkout(tmp_path / "work", "9.9.9")
        (unrelated / "notes").mkdir()

        beside = resolve_stdlib_root(home=home, env={}, anchor=unrelated / "notes" / "main.agl")
        at_the_root = resolve_stdlib_root(home=home, env={}, anchor=unrelated)

        assert beside == store_stdlib.resolve()
        assert at_the_root == store_stdlib.resolve()
