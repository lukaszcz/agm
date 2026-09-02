"""Tests for RootSet and assemble_roots in src/agm/agl/modules/roots.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.modules.roots import RootSet, assemble_roots
from agm.packages.manifest import load_manifest
from agm.packages.model import PackageInfo


def _package_with_a_rogue_module(tmp_path: Path, name: str = "demo") -> PackageInfo:
    """Mount a package whose root also holds a module outside its module tree."""
    root = tmp_path / "package"
    (root / name).mkdir(parents=True)
    (root / "package.toml").write_text(f'[package]\nname = "{name}"\nversion = "1.0.0"\n')
    (root / name / "main.agl").write_text("")
    (root / "assets").mkdir()
    (root / "assets" / "rogue.agl").write_text("")
    return PackageInfo(root, load_manifest(root / "package.toml"))


def _roots_for(tmp_path: Path, package: PackageInfo, *, loose: bool) -> RootSet:
    """Mount *package*, optionally also supplying its root as an ordinary root."""
    return assemble_roots(
        invocation_root=None,
        lib_root=None,
        configured=[],
        cli=[str(package.root)] if loose else [],
        cwd=tmp_path,
        package_roots=(package,),
    )


class TestRootSet:
    def test_construction(self, tmp_path: Path) -> None:
        roots: frozenset[Path] = frozenset([tmp_path])
        rs = RootSet(roots=roots)
        assert rs.roots == roots

    def test_frozen(self, tmp_path: Path) -> None:
        rs = RootSet(roots=frozenset([tmp_path]))
        with pytest.raises((AttributeError, TypeError)):
            setattr(rs, "roots", frozenset())

    def test_sorted_roots_is_deterministic(self, tmp_path: Path) -> None:
        dirs = [tmp_path / f"root{i}" for i in range(5)]
        for d in dirs:
            d.mkdir()
        # Provide paths in arbitrary order
        roots: frozenset[Path] = frozenset(dirs[i] for i in [3, 1, 4, 0, 2])
        rs = RootSet(roots=roots)
        result = rs.sorted_roots()
        assert result == tuple(sorted(dirs))

    def test_sorted_roots_returns_tuple(self, tmp_path: Path) -> None:
        rs = RootSet(roots=frozenset([tmp_path]))
        assert isinstance(rs.sorted_roots(), tuple)

    def test_sorted_roots_empty(self) -> None:
        rs = RootSet(roots=frozenset())
        assert rs.sorted_roots() == ()


class TestMountScoping:
    """A package-only root admits only its own module tree; a loose root admits all."""

    def test_package_only_root_is_scoped_to_its_declared_module_segment(
        self, tmp_path: Path
    ) -> None:
        package = _package_with_a_rogue_module(tmp_path)
        roots = _roots_for(tmp_path, package, loose=False)

        assert roots.sorted_roots_for(("demo",)) == (package.root,)
        assert roots.sorted_roots_for(("assets",)) == ()

    def test_package_only_root_rejects_a_file_outside_its_module_tree(self, tmp_path: Path) -> None:
        package = _package_with_a_rogue_module(tmp_path)
        roots = _roots_for(tmp_path, package, loose=False)

        assert roots.admits_path(package.root, package.module_root / "main.agl")
        assert not roots.admits_path(package.root, package.root / "assets" / "rogue.agl")

    def test_supplying_the_same_path_as_an_ordinary_root_keeps_it_loose(
        self, tmp_path: Path
    ) -> None:
        package = _package_with_a_rogue_module(tmp_path)
        roots = _roots_for(tmp_path, package, loose=True)

        assert roots.sorted_roots_for(("assets",)) == (package.root,)
        assert roots.admits_path(package.root, package.root / "assets" / "rogue.agl")

    def test_a_root_with_no_package_mount_admits_everything(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        roots = assemble_roots(
            invocation_root=plain,
            lib_root=None,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )

        assert roots.sorted_roots_for(("anything",)) == (plain.resolve(),)
        assert roots.admits_path(plain.resolve(), plain / "anything.agl")


class TestLooseRootsAreDistinguishing:
    def test_root_sets_differing_only_in_loose_roots_are_not_equal(self, tmp_path: Path) -> None:
        # The two root sets search the same directories, carry the same package,
        # and share the same standard library -- they differ only in whether the
        # package root is also loose, which is exactly what decides the files
        # they expose.  Anything that identifies a root set (a cached compiled
        # image, say) must therefore see them as different.
        package = _package_with_a_rogue_module(tmp_path)
        scoped = _roots_for(tmp_path, package, loose=False)
        loose = _roots_for(tmp_path, package, loose=True)

        assert scoped.roots == loose.roots
        assert scoped.packages == loose.packages
        assert scoped.stdlib_roots == loose.stdlib_roots
        assert scoped.loose_roots != loose.loose_roots
        assert scoped != loose


class TestAssembleRoots:
    def test_includes_invocation_root(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        assert inv_root.resolve() in rs.roots

    def test_includes_lib_root(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        lib = tmp_path / "lib"
        lib.mkdir()
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=lib,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        assert lib.resolve() in rs.roots

    def test_includes_stdlib_root(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        stdlib = tmp_path / "stdlib"
        stdlib.mkdir()
        rs = assemble_roots(
            invocation_root=inv_root,
            stdlib_root=stdlib,
            lib_root=None,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        assert stdlib.resolve() in rs.roots
        assert rs.stdlib_roots == frozenset({stdlib.resolve()})
        assert rs.is_standard_library_path(stdlib / "std" / "prelude.agl")

    def test_includes_package_root_and_preserves_its_ownership_metadata(
        self, tmp_path: Path
    ) -> None:
        invocation_root = tmp_path / "inv"
        invocation_root.mkdir()
        package_root = tmp_path / "package"
        (package_root / "demo").mkdir(parents=True)
        (package_root / "package.toml").write_text('[package]\nname = "demo"\nversion = "1.0.0"\n')
        package = PackageInfo(package_root, load_manifest(package_root / "package.toml"))

        roots = assemble_roots(
            invocation_root=invocation_root,
            lib_root=None,
            configured=[],
            cli=[],
            cwd=tmp_path,
            package_roots=(package,),
        )

        assert package.root in roots.roots
        assert roots.packages == (package,)

    def test_missing_package_root_is_not_mounted(self, tmp_path: Path) -> None:
        invocation_root = tmp_path / "inv"
        invocation_root.mkdir()
        manifest_root = tmp_path / "manifest"
        manifest_root.mkdir()
        manifest_path = manifest_root / "package.toml"
        manifest_path.write_text('[package]\nname = "demo"\nversion = "1.0.0"\n')
        package = PackageInfo(tmp_path / "missing", load_manifest(manifest_path))

        roots = assemble_roots(
            invocation_root=invocation_root,
            lib_root=None,
            configured=[],
            cli=[],
            cwd=tmp_path,
            package_roots=(package,),
        )

        assert package.root not in roots.roots
        assert roots.packages == ()

    def test_none_lib_root_not_included(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        assert len(rs.roots) == 1

    def test_configured_relative_resolves_against_origin(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        origin = tmp_path / "config_home"
        origin.mkdir()
        extra = origin / "mylib"
        extra.mkdir()
        # raw path is relative, should resolve against origin
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=[("mylib", origin)],
            cli=[],
            cwd=tmp_path,
        )
        assert extra.resolve() in rs.roots

    def test_configured_absolute_path(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        origin = tmp_path / "config_home"
        origin.mkdir()
        extra = tmp_path / "absolute_lib"
        extra.mkdir()
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=[(str(extra), origin)],
            cli=[],
            cwd=tmp_path,
        )
        assert extra.resolve() in rs.roots

    def test_cli_path_resolves_against_cwd(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        cli_lib = tmp_path / "cli_lib"
        cli_lib.mkdir()
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=[],
            cli=["cli_lib"],  # relative — resolves against cwd=tmp_path
            cwd=tmp_path,
        )
        assert cli_lib.resolve() in rs.roots

    def test_cli_absolute_path(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        cli_lib = tmp_path / "cli_abs"
        cli_lib.mkdir()
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=[],
            cli=[str(cli_lib)],
            cwd=tmp_path,
        )
        assert cli_lib.resolve() in rs.roots

    def test_tilde_expansion_in_lib_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A fake home keeps the expansion checkable: the directory `~` names
        # really exists, so the expanded root must be mounted rather than
        # silently dropped as missing.
        home = tmp_path / "home"
        lib = home / ".agm" / "lib"
        lib.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        inv_root = tmp_path / "inv"
        inv_root.mkdir()

        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=Path("~/.agm/lib"),
            configured=[],
            cli=[],
            cwd=tmp_path,
        )

        assert lib.resolve() in rs.roots
        for root in rs.roots:
            assert "~" not in str(root)

    def test_tilde_expansion_in_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        configured_lib = home / "agl-lib"
        configured_lib.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        inv_root = tmp_path / "inv"
        inv_root.mkdir()

        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=[("~/agl-lib", tmp_path)],
            cli=[],
            cwd=tmp_path,
        )

        assert configured_lib.resolve() in rs.roots
        for root in rs.roots:
            assert "~" not in str(root)

    def test_tilde_path_that_does_not_exist_under_home_is_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        inv_root = tmp_path / "inv"
        inv_root.mkdir()

        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=[("~/absent-lib", tmp_path)],
            cli=["~/absent-cli-lib"],
            cwd=tmp_path,
        )

        assert rs.roots == frozenset({inv_root.resolve()})

    def test_nonexistent_roots_dropped_silently(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        missing = tmp_path / "does_not_exist"
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=missing,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        assert missing.resolve() not in rs.roots
        # inv_root still included
        assert inv_root.resolve() in rs.roots

    def test_duplicate_roots_deduped(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        # Pass same path as both lib_root and cli
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=inv_root,
            configured=[],
            cli=[str(inv_root)],
            cwd=tmp_path,
        )
        # Should have only one canonical entry for inv_root
        assert len(rs.roots) == 1

    def test_symlinked_roots_collapsed_to_one(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        rs = assemble_roots(
            invocation_root=real,
            lib_root=link,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        # real and link resolve to the same canonical path — only one entry
        assert len(rs.roots) == 1
        assert real.resolve() in rs.roots

    def test_nested_roots_are_distinct(self, tmp_path: Path) -> None:
        """Nested roots (parent and child dirs) are NOT collapsed — distinct canonical paths."""
        parent = tmp_path / "root"
        parent.mkdir()
        child = parent / "sub"
        child.mkdir()
        rs = assemble_roots(
            invocation_root=parent,
            lib_root=child,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        assert len(rs.roots) == 2

    def test_configured_multiple_paths_different_origins(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        origin_a = tmp_path / "config_a"
        origin_a.mkdir()
        origin_b = tmp_path / "config_b"
        origin_b.mkdir()
        lib_a = origin_a / "mylib"
        lib_a.mkdir()
        lib_b = origin_b / "mylib"
        lib_b.mkdir()
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=[
                ("mylib", origin_a),
                ("mylib", origin_b),
            ],
            cli=[],
            cwd=tmp_path,
        )
        # Both resolve to different canonical paths
        assert lib_a.resolve() in rs.roots
        assert lib_b.resolve() in rs.roots

    def test_configured_escaped_percent_hole_survives_single_interpolation_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: a ``\\%{...}`` escape in ``[modules] roots`` is resolved to a
        literal ``%{...}`` by :func:`load_module_roots` alone.  ``assemble_roots``
        must not interpolate the already-resolved value a second time, or the
        escape is defeated and the wrong directory is searched.
        """
        from agm.config.module_roots import load_module_roots

        monkeypatch.setenv("TEAM", "alpha")
        home = tmp_path / "home"
        config_dir = home / ".agm"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text('[modules]\nroots = ["\\\\%{TEAM}/lib"]\n')

        # The literally-named directory the escape is meant to preserve, and the
        # directory a wrongful second interpolation pass would resolve to instead.
        literal_root = config_dir / "%{TEAM}" / "lib"
        literal_root.mkdir(parents=True)
        wrongly_expanded_root = config_dir / "alpha" / "lib"
        wrongly_expanded_root.mkdir(parents=True)

        mr_config = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=mr_config.extra,
            cli=[],
            cwd=tmp_path,
        )

        assert literal_root.resolve() in rs.roots
        assert wrongly_expanded_root.resolve() not in rs.roots

    def test_configured_env_value_containing_percent_hole_not_expanded_twice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: when an env value substituted into ``[modules] roots`` by
        :func:`load_module_roots` itself contains ``%{...}`` syntax, that syntax
        must remain literal — ``assemble_roots`` must not scan and expand it
        again against the environment.
        """
        from agm.config.module_roots import load_module_roots

        monkeypatch.setenv("LIB_BASE", str(tmp_path / "srv" / "%{TEAM}"))
        monkeypatch.setenv("TEAM", "core")
        home = tmp_path / "home"
        config_dir = home / ".agm"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text('[modules]\nroots = ["%{LIB_BASE}/agl"]\n')

        literal_root = tmp_path / "srv" / "%{TEAM}" / "agl"
        literal_root.mkdir(parents=True)
        wrongly_expanded_root = tmp_path / "srv" / "core" / "agl"
        wrongly_expanded_root.mkdir(parents=True)

        mr_config = load_module_roots(home=home, proj_dir=None, cwd=tmp_path)
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=mr_config.extra,
            cli=[],
            cwd=tmp_path,
        )

        assert literal_root.resolve() in rs.roots
        assert wrongly_expanded_root.resolve() not in rs.roots

    def test_all_roots_are_absolute(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        for root in rs.roots:
            assert root.is_absolute()

    def test_roots_are_canonical(self, tmp_path: Path) -> None:
        """All roots should be canonical (resolved, no symlinks in path)."""
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        rs = assemble_roots(
            invocation_root=link,
            lib_root=None,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        assert real.resolve() in rs.roots

    def test_returns_rootset(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        result = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        assert isinstance(result, RootSet)

    def test_invocation_root_nonexistent_dropped(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing_inv"
        rs = assemble_roots(
            invocation_root=missing,
            lib_root=None,
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        assert len(rs.roots) == 0
