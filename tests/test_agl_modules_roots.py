"""Tests for RootSet and assemble_roots in src/agm/agl/modules/roots.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.modules.roots import RootSet, assemble_roots


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

    def test_tilde_expansion_in_lib_root(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        # Use a real tilde path — we can only test expansion doesn't break things
        # We won't assert it exists (it may not), just that it's expanded + canonical
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=Path("~/.agm/lib"),
            configured=[],
            cli=[],
            cwd=tmp_path,
        )
        for root in rs.roots:
            assert "~" not in str(root)

    def test_tilde_expansion_in_configured(self, tmp_path: Path) -> None:
        inv_root = tmp_path / "inv"
        inv_root.mkdir()
        origin = tmp_path
        rs = assemble_roots(
            invocation_root=inv_root,
            lib_root=None,
            configured=[("~/nonexistent_agm_test_lib_xyz", origin)],
            cli=[],
            cwd=tmp_path,
        )
        for root in rs.roots:
            assert "~" not in str(root)

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
